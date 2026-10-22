import torch
import torch.nn as nn
import json
import os
import time
from PIL import Image
try:
    import torchvision.transforms as T
except ImportError:
    T = None
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from loop_models.aggregators.salad import SALAD
from loop_models.backbones.dinov2 import DINOv2


class _SaladRuntime(nn.Module):
    """Inference-only equivalent of the upstream Lightning VPR wrapper."""

    def __init__(self, config):
        super().__init__()
        self.backbone = DINOv2(
            model_name="dinov2_vitb14",
            num_trainable_blocks=4,
            return_token=True,
            norm_layer=True,
            vggt_long_config=config,
        )
        self.aggregator = SALAD(
            num_channels=768,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
        )

    def forward(self, images):
        return self.aggregator(self.backbone(images))


class _LoopImageDataset(Dataset):
    def __init__(self, image_paths, transform, image_size):
        self.image_paths = list(image_paths)
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"Error processing image {path}: {e}")
            if self.image_size is None:
                return torch.zeros(3, 224, 224)
            return torch.zeros(3, self.image_size[0], self.image_size[1])

LOOP_PROPOSAL_SCHEMA_VERSION = 1


def _safe_cache_component(value, default="item"):
    text = str(value).strip()
    safe = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in text)
    safe = safe.strip("._-")
    return (safe or default)[:96]


class LoopDetector:
    """Loop detector class for detecting loop closures in image sequences"""

    def __init__(self, image_dir,
                 output="loop_closures.txt",
                 config=None,
                 proposal_sidecar_enabled=False,
                 proposal_sidecar_path=None,
                 proposal_universe_enabled=False):
        """Initialize the loop detector

        Args:
            image_dir: Directory path containing images
            ckpt_path: Model checkpoint path
            image_size: Image resize dimensions [height width]
            batch_size: Batch size for processing
            similarity_threshold: Similarity threshold for loop closure
            top_k: Number of nearest neighbors to check for each image
            use_nms: Whether to use Non-Maximum Suppression (NMS) filtering
            nms_threshold: NMS threshold for minimum frame difference between loop pairs
            output: Output file path
        """
        self.config = config
        self.image_dir = image_dir
        self.ckpt_path = self.config['Weights']['SALAD']
        self.image_size = self.config['Loop']['SALAD']['image_size']
        self.batch_size = self.config['Loop']['SALAD']['batch_size']
        self.similarity_threshold = self.config['Loop']['SALAD']['similarity_threshold']
        self.top_k = self.config['Loop']['SALAD']['top_k']
        self.use_nms = self.config['Loop']['SALAD']['use_nms']
        self.nms_threshold = self.config['Loop']['SALAD']['nms_threshold']
        self.output = output

        self.model = None
        self.device = None
        self.image_paths = None
        self.descriptors = None
        self.loop_closures = None
        self.proposal_sidecar_enabled = bool(proposal_sidecar_enabled)
        self.proposal_sidecar_path = proposal_sidecar_path
        self.proposal_universe_enabled = bool(proposal_universe_enabled)
        self.loop_proposal_records = []
        self.raw_loop_proposals = []
        self._raw_loop_proposal_order = []
        self._raw_loop_proposal_by_candidate = {}
        self._loop_metadata_by_candidate = {}

    def _descriptor_cache_config(self):
        cache_config = self.config.get('Loop', {}).get('SALAD', {}).get('cache', {})
        if isinstance(cache_config, bool):
            cache_config = {'enable': cache_config}
        return cache_config or {}

    def _cache_stat(self, path):
        try:
            stat = Path(path).stat()
        except OSError:
            return {"path": str(path), "missing": True}
        return {
            "path": str(Path(path).resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def _descriptor_cache_entry(self):
        cache_config = self._descriptor_cache_config()
        if not cache_config.get('enable', False):
            return None
        root = cache_config.get('root')
        if not root:
            return None
        if self.image_paths is None:
            self.get_image_paths()

        root = Path(os.path.expandvars(os.path.expanduser(str(root)))).resolve()
        key_payload = {
            "version": cache_config.get('key_version', 'v1'),
            "image_dir": str(Path(self.image_dir).resolve()),
            "image_size": self.image_size,
            "model": "dinov2_vitb14_salad",
            "salad": self._cache_stat(self.ckpt_path),
            "dinov2": self._cache_stat(self.config['Weights']['DNIO']),
            "images": [self._cache_stat(path) for path in self.image_paths],
        }
        image_root = Path(key_payload["image_dir"])
        image_parts = [
            _safe_cache_component(part, "root")
            for part in image_root.parts
            if part not in ("", image_root.anchor)
        ]
        if not image_parts:
            image_parts = ["root"]
        image_size = self.image_size or ["native", "native"]
        size_label = "x".join(str(part) for part in image_size)
        cache_label = (
            f"{_safe_cache_component(cache_config.get('key_version', 'v1'))}"
            f"__frames_{len(self.image_paths):06d}"
            f"__size_{_safe_cache_component(size_label)}"
        )
        cache_dir = root / "dinov2_vitb14_salad" / Path(*image_parts)
        return {
            "path": cache_dir / f"{cache_label}.pt",
            "metadata_path": cache_dir / f"{cache_label}.json",
            "lock_dir": cache_dir / f"{cache_label}.lock",
            "payload": key_payload,
            "cache_label": cache_label,
            "wait_seconds": float(cache_config.get('wait_seconds', 0) or 0),
            "stale_lock_seconds": float(cache_config.get('stale_lock_seconds', 7200) or 7200),
        }

    def _descriptor_cache_metadata_matches(self, entry):
        try:
            metadata = json.loads(entry["metadata_path"].read_text())
        except FileNotFoundError:
            print(f"SALAD descriptor cache ignored because metadata is missing: {entry['metadata_path']}")
            return False
        except Exception as exc:
            print(f"SALAD descriptor cache ignored because metadata could not be read: {entry['metadata_path']} ({exc})")
            return False
        if metadata.get("payload") != entry["payload"]:
            print(f"SALAD descriptor cache ignored because metadata does not match current inputs: {entry['metadata_path']}")
            return False
        return True

    def _load_descriptor_cache(self, entry):
        if entry is None or not entry["path"].exists():
            return None
        if not self._descriptor_cache_metadata_matches(entry):
            return None
        try:
            try:
                descriptors = torch.load(entry["path"], map_location="cpu", weights_only=True)
            except TypeError:
                descriptors = torch.load(entry["path"], map_location="cpu")
        except Exception as exc:
            print(f"SALAD descriptor cache load failed, recomputing: {entry['path']} ({exc})")
            return None
        if not isinstance(descriptors, torch.Tensor):
            print(f"SALAD descriptor cache ignored because payload is not a tensor: {entry['path']}")
            return None
        if self.image_paths is not None and descriptors.shape[0] != len(self.image_paths):
            print(
                "SALAD descriptor cache ignored because frame count mismatches: "
                f"{entry['path']} ({descriptors.shape[0]} vs {len(self.image_paths)})"
            )
            return None
        print(f"SALAD descriptor cache hit: {entry['path']}")
        return descriptors

    def _wait_for_descriptor_cache(self, entry):
        if entry is None or entry["wait_seconds"] <= 0:
            return None
        deadline = time.time() + entry["wait_seconds"]
        while time.time() < deadline:
            descriptors = self._load_descriptor_cache(entry)
            if descriptors is not None:
                return descriptors
            if not entry["lock_dir"].exists():
                return None
            time.sleep(5.0)
        return None

    def _acquire_descriptor_cache_lock(self, entry):
        if entry is None:
            return False
        try:
            entry["path"].parent.mkdir(parents=True, exist_ok=True)
            entry["lock_dir"].mkdir()
            (entry["lock_dir"] / "pid").write_text(str(os.getpid()))
            (entry["lock_dir"] / "created_at").write_text(str(time.time()))
            return True
        except FileExistsError:
            try:
                created_at = float((entry["lock_dir"] / "created_at").read_text().strip())
            except Exception:
                created_at = time.time()
            if time.time() - created_at > entry["stale_lock_seconds"]:
                try:
                    for child in entry["lock_dir"].iterdir():
                        child.unlink()
                    entry["lock_dir"].rmdir()
                    return self._acquire_descriptor_cache_lock(entry)
                except Exception:
                    return False
            return False

    def _release_descriptor_cache_lock(self, entry):
        if entry is None:
            return
        try:
            for child in entry["lock_dir"].iterdir():
                child.unlink()
            entry["lock_dir"].rmdir()
        except FileNotFoundError:
            return
        except Exception as exc:
            print(f"SALAD descriptor cache lock cleanup failed: {entry['lock_dir']} ({exc})")

    def _save_descriptor_cache(self, entry, descriptors):
        if entry is None:
            return
        tmp_path = entry["path"].with_suffix(f".{os.getpid()}.tmp")
        try:
            entry["path"].parent.mkdir(parents=True, exist_ok=True)
            torch.save(descriptors.cpu(), tmp_path)
            os.replace(tmp_path, entry["path"])
            metadata = {
                "cache_label": entry["cache_label"],
                "created_at": time.time(),
                "num_images": len(self.image_paths or []),
                "shape": list(descriptors.shape),
                "payload": entry["payload"],
            }
            entry["metadata_path"].write_text(json.dumps(metadata, indent=2, sort_keys=True))
            print(f"SALAD descriptor cache saved: {entry['path']}")
        except Exception as exc:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            print(f"SALAD descriptor cache write failed; continuing without cache: {entry['path']} ({exc})")

    def _input_transform(self, image_size=None):
        """Create image transformation function"""
        if T is None:
            raise ImportError("torchvision is required for SALAD image transforms")
        MEAN=[0.485, 0.456, 0.406]; STD=[0.229, 0.224, 0.225]
        if image_size:
            return T.Compose([
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
        else:
            return T.Compose([
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])

    def load_model(self):
        """Load model"""
        model = _SaladRuntime(self.config)

        model.load_state_dict(torch.load(self.ckpt_path))
        model = model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Model loaded: {self.ckpt_path}")

        self.model = model
        self.device = device
        return model, device

    def get_image_paths(self):
        """Get paths of all image files in directory"""
        image_extensions = [".jpg", ".jpeg", ".png"]
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(list(Path(self.image_dir).glob(f"*{ext}")))
            image_paths.extend(list(Path(self.image_dir).glob(f"*{ext.upper()}")))

        image_paths = sorted(image_paths)
        self.image_paths = image_paths
        return image_paths

    def extract_descriptors(self):
        """Extract image feature descriptors"""
        if self.image_paths is None:
            self.get_image_paths()

        cache_entry = self._descriptor_cache_entry()
        cached_descriptors = self._load_descriptor_cache(cache_entry)
        if cached_descriptors is None:
            lock_acquired = self._acquire_descriptor_cache_lock(cache_entry)
            if not lock_acquired:
                cached_descriptors = self._wait_for_descriptor_cache(cache_entry)
        else:
            lock_acquired = False

        if cached_descriptors is not None:
            self.descriptors = cached_descriptors
            return self.descriptors

        if self.model is None or self.device is None:
            self.load_model()

        transform = self._input_transform(self.image_size)
        descriptors = []

        num_workers = int(self.config['Loop']['SALAD'].get('num_workers', 0) or 0)
        if num_workers > 0:
            pin_memory = bool(self.config['Loop']['SALAD'].get('pin_memory', True))
            prefetch_factor = int(self.config['Loop']['SALAD'].get('prefetch_factor', 2) or 2)
            persistent_workers = bool(self.config['Loop']['SALAD'].get('persistent_workers', True))
            non_blocking = bool(self.config['Loop']['SALAD'].get('non_blocking', True))
            dataset = _LoopImageDataset(self.image_paths, transform, self.image_size)
            loader_kwargs = {
                'batch_size': self.batch_size,
                'shuffle': False,
                'num_workers': num_workers,
                'pin_memory': pin_memory,
                'persistent_workers': persistent_workers,
                'prefetch_factor': prefetch_factor,
            }

            for batch_tensor in tqdm(DataLoader(dataset, **loader_kwargs), desc="Extracting features"):
                batch_tensor = batch_tensor.to(self.device, non_blocking=non_blocking)
                with torch.no_grad():
                    with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                        batch_descriptors = self.model(batch_tensor).cpu()
                descriptors.append(batch_descriptors)

            self.descriptors = torch.cat(descriptors)
            if lock_acquired:
                try:
                    self._save_descriptor_cache(cache_entry, self.descriptors)
                finally:
                    self._release_descriptor_cache_lock(cache_entry)
            return self.descriptors

        for i in tqdm(range(0, len(self.image_paths), self.batch_size), desc="Extracting features"):
            batch_paths = self.image_paths[i:i+self.batch_size]
            batch_imgs = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img = transform(img)
                    batch_imgs.append(img)
                except Exception as e:
                    print(f"Error processing image {path}: {e}")
                    img = torch.zeros(3, 224, 224) if self.image_size is None else torch.zeros(3, self.image_size[0], self.image_size[1])
                    batch_imgs.append(img)

            batch_tensor = torch.stack(batch_imgs).to(self.device)

            with torch.no_grad():
                with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                    batch_descriptors = self.model(batch_tensor).cpu()

            descriptors.append(batch_descriptors)

        self.descriptors = torch.cat(descriptors)
        if lock_acquired:
            try:
                self._save_descriptor_cache(cache_entry, self.descriptors)
            finally:
                self._release_descriptor_cache_lock(cache_entry)
        return self.descriptors

    @staticmethod
    def _candidate_id(idx1, idx2):
        frame_a = int(idx1)
        frame_b = int(idx2)
        low = min(frame_a, frame_b)
        high = max(frame_a, frame_b)
        return f"salad:{low:06d}:{high:06d}"

    @staticmethod
    def _candidate_frames(idx1, idx2):
        frame_a = int(idx1)
        frame_b = int(idx2)
        return [min(frame_a, frame_b), max(frame_a, frame_b)]

    def _proposal_event_id(self, stage, status, idx1, idx2, metadata, event_index):
        parts = [
            "proposal_event",
            f"{int(event_index):06d}",
            self._candidate_id(idx1, idx2),
            _safe_cache_component(stage),
            _safe_cache_component(status),
        ]
        for key in ("query_frame", "neighbor_frame", "neighbor_rank"):
            if metadata.get(key) is not None:
                parts.append(f"{key}_{metadata[key]}")
        return ":".join(str(part) for part in parts)

    def _proposal_config_summary(self):
        config = self.config or {}
        loop_config = config.get("Loop", {}).get("SALAD", {})
        weights = config.get("Weights", {})
        return {
            "loop_salad": json.loads(json.dumps(loop_config, sort_keys=True, default=str)),
            "weights": {
                key: str(weights[key])
                for key in ("SALAD", "DNIO")
                if key in weights
            },
        }

    def _record_loop_proposal(self, stage, status, idx1, idx2, similarity, drop_reason=None, **metadata):
        if not self.proposal_sidecar_enabled:
            return
        record = {
            "schema_version": LOOP_PROPOSAL_SCHEMA_VERSION,
            "record_type": "loop_proposal",
            "detector": "SALAD",
            "candidate_id": self._candidate_id(idx1, idx2),
            "frames": self._candidate_frames(idx1, idx2),
            "similarity": float(similarity),
            "stage": str(stage),
            "status": str(status),
            "drop_reason": drop_reason,
        }
        if self.proposal_universe_enabled:
            event_index = len(self.loop_proposal_records)
            record.update({
                "table": "proposal_event",
                "proposal_event_id": self._proposal_event_id(
                    stage,
                    status,
                    idx1,
                    idx2,
                    metadata,
                    event_index,
                ),
                "frame_pair_candidate_id": record["candidate_id"],
                "candidate_scope": "frame_pair",
                "loop_exists_target": None,
                "config_summary": self._proposal_config_summary(),
                "proposal_universe_schema": f"e226_proposal_universe_v{LOOP_PROPOSAL_SCHEMA_VERSION}",
            })
            if status == "dropped":
                record["drop_is_negative"] = False
            if drop_reason == "duplicate_pair":
                record["dedup_drop_is_negative"] = False
        for key, value in metadata.items():
            if value is None:
                record[key] = None
            elif isinstance(value, (int, float, str, bool, list, dict)):
                record[key] = value
            else:
                try:
                    record[key] = value.item()
                except AttributeError:
                    record[key] = str(value)
        self.loop_proposal_records.append(record)

    def _loop_metadata(self, idx1, idx2, similarity):
        return {
            "proposal_schema_version": LOOP_PROPOSAL_SCHEMA_VERSION,
            "candidate_id": self._candidate_id(idx1, idx2),
            "proposal_frames": self._candidate_frames(idx1, idx2),
            "similarity": float(similarity),
        }

    def _observe_raw_loop_proposal(self, idx1, idx2, similarity, **metadata):
        candidate_id = self._candidate_id(idx1, idx2)
        frames = self._candidate_frames(idx1, idx2)
        representative = {
            "idx1": int(max(idx1, idx2)),
            "idx2": int(min(idx1, idx2)),
            "similarity": float(similarity),
            "metadata": {
                "proposal_schema_version": LOOP_PROPOSAL_SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "proposal_frames": frames,
                "similarity": float(similarity),
                "detector": "SALAD",
                "pre_admission": True,
            },
        }
        for key, value in metadata.items():
            if value is None:
                representative["metadata"][key] = None
            elif isinstance(value, (int, float, str, bool, list, dict)):
                representative["metadata"][key] = value
            else:
                try:
                    representative["metadata"][key] = value.item()
                except AttributeError:
                    representative["metadata"][key] = str(value)

        existing = self._raw_loop_proposal_by_candidate.get(candidate_id)
        if existing is None:
            representative["metadata"]["raw_canonical_rank"] = len(self._raw_loop_proposal_order)
            representative["metadata"]["raw_observation_count"] = 1
            self._raw_loop_proposal_order.append(candidate_id)
            self._raw_loop_proposal_by_candidate[candidate_id] = representative
            return

        observation_count = int(existing["metadata"].get("raw_observation_count", 1)) + 1
        canonical_rank = int(existing["metadata"].get("raw_canonical_rank", 0))
        keep_new = float(representative["similarity"]) > float(existing["similarity"])
        if keep_new:
            representative["metadata"]["raw_canonical_rank"] = canonical_rank
            representative["metadata"]["raw_observation_count"] = observation_count
            representative["metadata"]["raw_duplicate_observed"] = True
            self._raw_loop_proposal_by_candidate[candidate_id] = representative
        else:
            existing["metadata"]["raw_observation_count"] = observation_count
            existing["metadata"]["raw_duplicate_observed"] = True

    def _finalize_raw_loop_proposals(self):
        self.raw_loop_proposals = [
            self._raw_loop_proposal_by_candidate[candidate_id]
            for candidate_id in self._raw_loop_proposal_order
            if candidate_id in self._raw_loop_proposal_by_candidate
        ]

    def get_raw_loop_proposal_list(self, include_similarity=False, include_metadata=False):
        result = []
        for proposal in self.raw_loop_proposals or []:
            idx1 = int(proposal["idx1"])
            idx2 = int(proposal["idx2"])
            similarity = float(proposal["similarity"])
            metadata = dict(proposal.get("metadata") or {})
            if include_metadata:
                if include_similarity:
                    result.append((idx1, idx2, similarity, metadata))
                else:
                    result.append((idx1, idx2, metadata))
            elif include_similarity:
                result.append((idx1, idx2, similarity))
            else:
                result.append((idx1, idx2))
        return result

    def write_loop_proposals(self, output_path=None):
        if not self.proposal_sidecar_enabled:
            return None
        path = output_path or self.proposal_sidecar_path
        if not path:
            return None
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for record in self.loop_proposal_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"Loop proposal sidecar saved to {path}")
        return path

    def _apply_nms_filter(self, loop_closures, nms_threshold, record_proposals=False):
        """Apply Non-Maximum Suppression (NMS) filtering to loop pairs"""
        if not loop_closures or nms_threshold <= 0:
            if record_proposals:
                for idx1, idx2, sim in loop_closures:
                    self._record_loop_proposal(
                        "nms",
                        "kept",
                        idx1,
                        idx2,
                        sim,
                        nms_threshold=int(nms_threshold),
                    )
            return loop_closures

        sorted_loops = sorted(loop_closures, key=lambda x: x[2], reverse=True)
        filtered_loops = []
        suppressed = set()

        max_frame = max(max(idx1, idx2) for idx1, idx2, _ in loop_closures)

        for idx1, idx2, sim in sorted_loops:
            if idx1 in suppressed or idx2 in suppressed:
                if record_proposals:
                    self._record_loop_proposal(
                        "nms",
                        "dropped",
                        idx1,
                        idx2,
                        sim,
                        drop_reason="nms_suppressed",
                        nms_threshold=int(nms_threshold),
                    )
                continue

            filtered_loops.append((idx1, idx2, sim))
            if record_proposals:
                self._record_loop_proposal(
                    "nms",
                    "kept",
                    idx1,
                    idx2,
                    sim,
                    nms_threshold=int(nms_threshold),
                )

            suppress_range = set()

            start1 = max(0, idx1 - nms_threshold)
            end1 = min(idx1 + nms_threshold + 1, idx2)
            suppress_range.update(range(start1, end1))

            start2 = max(idx1 + 1, idx2 - nms_threshold)
            end2 = min(idx2 + nms_threshold + 1, max_frame + 1)
            suppress_range.update(range(start2, end2))

            suppressed.update(suppress_range)

        return filtered_loops

    def _ensure_decending_order(self, tuples_list):
        return [(max(a, b), min(a, b), score) for a, b, score in tuples_list]

    def find_loop_closures(self):
        """Find loop closures"""
        self.loop_proposal_records = []
        self.raw_loop_proposals = []
        self._raw_loop_proposal_order = []
        self._raw_loop_proposal_by_candidate = {}
        self._loop_metadata_by_candidate = {}
        if self.descriptors is None:
            self.extract_descriptors()

        import faiss

        embed_size = self.descriptors.shape[1]
        faiss_index = faiss.IndexFlatIP(embed_size)

        normalized_descriptors = self.descriptors.numpy()
        faiss_index.add(normalized_descriptors)

        similarities, indices = faiss_index.search(normalized_descriptors, self.top_k + 1)  # +1 because self is most similar

        loop_closures = []
        for i in range(len(self.descriptors)):
            # Skip first result (self)
            for j in range(1, self.top_k + 1):
                neighbor_idx = indices[i, j]
                similarity = similarities[i, j]
                temporal_gap_frames = int(abs(i - neighbor_idx))
                if similarity <= self.similarity_threshold:
                    pre_admission_status = "dropped"
                    pre_admission_drop_reason = "below_similarity_threshold"
                elif temporal_gap_frames <= 10:
                    pre_admission_status = "dropped"
                    pre_admission_drop_reason = "temporal_gap"
                else:
                    pre_admission_status = "kept"
                    pre_admission_drop_reason = None
                self._observe_raw_loop_proposal(
                    i,
                    neighbor_idx,
                    similarity,
                    query_frame=int(i),
                    neighbor_frame=int(neighbor_idx),
                    neighbor_rank=int(j),
                    pre_admission_stage="threshold_temporal_gap",
                    pre_admission_status=pre_admission_status,
                    pre_admission_drop_reason=pre_admission_drop_reason,
                    similarity_threshold=float(self.similarity_threshold),
                    temporal_gap_frames=temporal_gap_frames,
                    min_temporal_gap_frames=10,
                )
                self._record_loop_proposal(
                    "raw_top_k",
                    "observed",
                    i,
                    neighbor_idx,
                    similarity,
                    query_frame=int(i),
                    neighbor_frame=int(neighbor_idx),
                    neighbor_rank=int(j),
                )

                if similarity <= self.similarity_threshold:
                    self._record_loop_proposal(
                        "threshold_temporal_gap",
                        "dropped",
                        i,
                        neighbor_idx,
                        similarity,
                        drop_reason="below_similarity_threshold",
                        similarity_threshold=float(self.similarity_threshold),
                        temporal_gap_frames=temporal_gap_frames,
                        min_temporal_gap_frames=10,
                    )
                    continue
                if abs(i - neighbor_idx) <= 10:
                    self._record_loop_proposal(
                        "threshold_temporal_gap",
                        "dropped",
                        i,
                        neighbor_idx,
                        similarity,
                        drop_reason="temporal_gap",
                        similarity_threshold=float(self.similarity_threshold),
                        temporal_gap_frames=temporal_gap_frames,
                        min_temporal_gap_frames=10,
                    )
                    continue

                self._record_loop_proposal(
                    "threshold_temporal_gap",
                    "kept",
                    i,
                    neighbor_idx,
                    similarity,
                    similarity_threshold=float(self.similarity_threshold),
                    temporal_gap_frames=temporal_gap_frames,
                    min_temporal_gap_frames=10,
                )
                if i < neighbor_idx:
                    loop_closures.append((i, neighbor_idx, similarity))
                else:
                    loop_closures.append((neighbor_idx, i, similarity))

        if self.proposal_sidecar_enabled:
            seen_candidates = set()
            for idx1, idx2, similarity in loop_closures:
                candidate_id = self._candidate_id(idx1, idx2)
                if candidate_id in seen_candidates:
                    self._record_loop_proposal(
                        "pair_dedup",
                        "dropped",
                        idx1,
                        idx2,
                        similarity,
                        drop_reason="duplicate_pair",
                    )
                else:
                    seen_candidates.add(candidate_id)
                    self._record_loop_proposal(
                        "pair_dedup",
                        "kept",
                        idx1,
                        idx2,
                        similarity,
                    )

        loop_closures = list(set(loop_closures))
        loop_closures.sort(key=lambda x: x[2], reverse=True)

        if self.use_nms and self.nms_threshold > 0:
            loop_closures = self._apply_nms_filter(
                loop_closures,
                self.nms_threshold,
                record_proposals=self.proposal_sidecar_enabled,
            )
        elif self.proposal_sidecar_enabled:
            for idx1, idx2, similarity in loop_closures:
                self._record_loop_proposal(
                    "nms",
                    "kept",
                    idx1,
                    idx2,
                    similarity,
                    nms_threshold=int(self.nms_threshold),
                    nms_enabled=False,
                )

        self.loop_closures = self._ensure_decending_order(loop_closures)
        if self.proposal_sidecar_enabled:
            for rank, (idx1, idx2, similarity) in enumerate(self.loop_closures):
                metadata = self._loop_metadata(idx1, idx2, similarity)
                metadata["detector_final_rank"] = int(rank)
                self._loop_metadata_by_candidate[metadata["candidate_id"]] = metadata
                self._record_loop_proposal(
                    "final_admitted",
                    "admitted",
                    idx1,
                    idx2,
                    similarity,
                    detector_final_rank=int(rank),
                )
        self._finalize_raw_loop_proposals()
        return self.loop_closures


    def save_results(self):
        """Save loop detection results to file"""
        if self.loop_closures is None:
            self.find_loop_closures()

        with open(self.output, 'w') as f:
            f.write("# Loop Detection Results (index1, index2, similarity)\n")
            if self.use_nms:
                f.write(f"# NMS filtering applied, threshold: {self.nms_threshold}\n")
            f.write("\n# Loop pairs:\n")
            for i, j, sim in self.loop_closures:
                f.write(f"{i}, {j}, {sim:.4f}\n")
            f.write("\n# Image path list:\n")
            for i, path in enumerate(self.image_paths):
                f.write(f"# {i}: {path}\n")

        print(f"Found {len(self.loop_closures)} loop pairs, results saved to {self.output}")
        if self.use_nms:
            print(f"NMS filtering applied, threshold: {self.nms_threshold}")

        if self.loop_closures:
            print("\nTop 10 loop pairs:")
            for i, (idx1, idx2, sim) in enumerate(self.loop_closures[:10]):
                print(f"{idx1}, {idx2}, similarity: {sim:.4f}")
                if i >= 9:
                    break

    def get_loop_list(self, include_similarity=False, include_metadata=False):
        if include_metadata:
            result = []
            for idx1, idx2, sim in self.loop_closures:
                metadata = dict(self._loop_metadata_by_candidate.get(
                    self._candidate_id(idx1, idx2),
                    {},
                ))
                if include_similarity:
                    result.append((idx1, idx2, float(sim), metadata))
                else:
                    result.append((idx1, idx2, metadata))
            return result
        if include_similarity:
            return [(idx1, idx2, float(sim)) for idx1, idx2, sim in self.loop_closures]
        return [(idx1, idx2) for idx1, idx2, _ in self.loop_closures]

    def run(self):
        """Run complete loop detection pipeline"""
        print('Loading model...')
        self.load_model()

        self.get_image_paths()
        if not self.image_paths:
            print(f"No image files found in {self.image_dir}")
            return

        print(f"Found {len(self.image_paths)} image files")

        self.extract_descriptors()

        self.find_loop_closures()

        self.save_results()
        self.write_loop_proposals()

        return self.loop_closures
