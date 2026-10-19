"""Independent instance extraction methods behind a common probability-map API."""
from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


@dataclass(frozen=True)
class PostprocessConfig:
    method: str = "boundary"
    foreground_threshold: float = .5
    boundary_threshold: float = .5
    min_area: int = 20
    seed_min_area: int = 3
    seed_distance: int = 5
    boundary_weight: float = 2.
    closing_radius: int = 0
    reconnect_distance: int = 0

    def __post_init__(self):
        if self.method not in ("components","boundary","watershed","continuity"): raise ValueError("Unknown extraction method")
        if not 0 < self.foreground_threshold < 1 or not 0 < self.boundary_threshold < 1: raise ValueError("Invalid threshold")
        if min(self.min_area,self.seed_min_area,self.seed_distance) < 1: raise ValueError("Areas and distances must be positive")
        if min(self.boundary_weight,self.closing_radius,self.reconnect_distance) < 0: raise ValueError("Negative morphology parameter")


def component_map(mask: np.ndarray) -> np.ndarray:
    return ndi.label(mask, structure=np.ones((3,3)))[0].astype(np.int32)


def _filter_labels(labels, min_area):
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False
    lookup = np.zeros(len(sizes),dtype=np.int32)
    lookup[keep] = np.arange(1,int(keep.sum())+1)
    return lookup[labels]


def _fallback_markers(markers, foreground):
    """An all-separation thin component still receives a seed, so it survives."""
    components = component_map(foreground)
    has_seed = np.unique(components[markers>0])
    missing = np.setdiff1d(np.unique(components), np.r_[0,has_seed])
    next_id = int(markers.max())
    # A full missing component can be one marker, preserving every narrow pixel.
    lookup = np.zeros(int(components.max())+1,dtype=np.int32)
    for key in missing:
        next_id += 1
        lookup[key] = next_id
    return np.where(markers>0,markers,lookup[components]).astype(np.int32)


def connected_components(fg, sep, cfg): return component_map(fg)


def boundary_components(fg, sep, cfg):
    markers = _filter_labels(component_map(fg & (sep < cfg.boundary_threshold)),cfg.seed_min_area)
    markers = _fallback_markers(markers,fg)
    # Geodesic refill allocates separator pixels to instances instead of deleting
    # them. Markers identify disconnected low-separation cores.
    return watershed(np.zeros(fg.shape,np.float32),markers,mask=fg,connectivity=np.ones((3,3))).astype(np.int32)


def marker_watershed(fg, sep, cfg):
    distance = ndi.distance_transform_edt(fg)
    core = fg & (sep < cfg.boundary_threshold)
    peaks = peak_local_max(distance, min_distance=cfg.seed_distance, labels=core.astype(np.uint8), exclude_border=False)
    markers = np.zeros(fg.shape,np.int32)
    if len(peaks): markers[tuple(peaks.T)] = np.arange(1,len(peaks)+1)
    markers = _fallback_markers(markers,fg)
    elevation = -distance + cfg.boundary_weight*sep
    return watershed(elevation,markers,mask=fg,connectivity=np.ones((3,3))).astype(np.int32)


def continuity_refinement(labels, foreground_probability, separation_probability, cfg):
    """Optional conservative gap union; no bridge pixels are added to masks.

    Merge only mutual-nearest, PCA-aligned elongated fragments with a short,
    foreground-supported, low-separation gap. Disconnected predicted regions can
    share an instance ID, just as a disconnected COCO annotation can.
    """
    if cfg.reconnect_distance <= 0: return labels
    from scipy.spatial import cKDTree
    objects = ndi.find_objects(labels)
    endpoints, owners, tangents = [], [], []
    for index, box in enumerate(objects,1):
        if box is None: continue
        ys,xs = np.nonzero(labels[box]==index)
        points = np.column_stack((ys+box[0].start,xs+box[1].start)).astype(float)
        if len(points)<3: continue
        values,vectors = np.linalg.eigh(np.cov(points.T))
        if values[-1] < 4*max(values[0],1): continue
        direction = vectors[:,-1]
        projections = points @ direction
        for i in (int(projections.argmin()),int(projections.argmax())):
            endpoints.append(points[i]); owners.append(index); tangents.append(direction)
    if len(endpoints)<2: return labels
    points=np.asarray(endpoints); tree=cKDTree(points)
    distances, neighbors=tree.query(points,k=min(4,len(points)))
    parent=np.arange(int(labels.max())+1)
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    chosen={}
    for i in range(len(points)):
        candidates=[int(j) for d,j in zip(np.atleast_1d(distances[i])[1:],np.atleast_1d(neighbors[i])[1:])
                    if d<=cfg.reconnect_distance and owners[int(j)]!=owners[i]]
        if candidates: chosen[i]=candidates[0]
    for i,j in chosen.items():
        if i>=j or chosen.get(j)!=i: continue
        delta=points[j]-points[i]; length=np.linalg.norm(delta)
        if length==0: continue
        direction=delta/length
        if min(abs(direction@tangents[i]),abs(direction@tangents[j])) < .95: continue
        line=np.rint(np.linspace(points[i],points[j],int(np.ceil(length))+1)).astype(int)
        ys,xs=line.T
        if (foreground_probability[ys,xs] < cfg.foreground_threshold*.8).any(): continue
        if (separation_probability[ys,xs] >= cfg.boundary_threshold*.5).any(): continue
        parent[find(owners[j])]=find(owners[i])
    lookup=np.array([find(i) for i in range(len(parent))],dtype=np.int32)
    return lookup[labels]


class InstanceMasks(Sequence):
    """Lazy independent binary masks backed by one ID map, bounding RAM usage.

    Iterate or index to materialize a single full-resolution mask. Explicitly
    calling list(...) is possible but can be expensive for fragmented predictions.
    """
    def __init__(self, labels): self.labels=labels; self.count=int(labels.max())
    def __len__(self): return self.count
    def __getitem__(self,index):
        if isinstance(index,slice): return [self[i] for i in range(*index.indices(self.count))]
        if index<0: index+=self.count
        if index<0 or index>=self.count: raise IndexError(index)
        return self.labels==index+1


def extract_instances(probabilities: np.ndarray, config=None) -> tuple[np.ndarray,Sequence]:
    """[2,H,W] probabilities -> ID map and independent HW boolean masks."""
    cfg=config if isinstance(config,PostprocessConfig) else PostprocessConfig(**(config or {}))
    p=np.asarray(probabilities)
    if p.ndim!=3 or p.shape[0]!=2 or not np.isfinite(p).all() or p.min()<0 or p.max()>1:
        raise ValueError("Expected finite [2,H,W] probabilities in [0,1]")
    fg=p[0]>=cfg.foreground_threshold
    if cfg.closing_radius:
        fg=ndi.binary_closing(fg,iterations=cfg.closing_radius) | fg
    if not fg.any(): return np.zeros(fg.shape,np.int32),[]
    method={"components":connected_components,"boundary":boundary_components,
            "watershed":marker_watershed,"continuity":boundary_components}[cfg.method]
    labels=method(fg,p[1],cfg)
    if cfg.method=="continuity": labels=continuity_refinement(labels,p[0],p[1],cfg)
    labels=_filter_labels(labels,cfg.min_area)
    return labels,InstanceMasks(labels) if labels.max() else []
