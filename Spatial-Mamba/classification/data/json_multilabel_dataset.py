import os, json
import torch
import torch.utils.data as data
from PIL import Image

class JSONMultiLabelDataset(data.Dataset):
    def __init__(self, root, ann_file, transform=None, classes=None, strict=True,
                return_group_id: bool=False):
        self.root = root
        self.ann = ann_file if os.path.isabs(ann_file) else os.path.join(root, ann_file)
        self.transform = transform
        self.db = json.load(open(self.ann, "r", encoding="utf-8"))
        self.strict = strict
        self.return_group_id = bool(return_group_id)

        if classes is None:
            names = set()
            for r in self.db:
                for n in r.get("labels", []):
                    names.add(n)
            self.class_names = sorted(names)
        else:
            self.class_names = list(classes)
        self.cls2id = {n: i for i, n in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        self._items = []
        for r in self.db:
            ipath = r["image_path"]
            ipath = ipath if os.path.isabs(ipath) else os.path.join(self.root, ipath)
            labs = r.get("labels", [])
            self._items.append((ipath, labs))
        # 20251121: construct group_id
        self._group_ids = []
        group2id = {}
        for ipath, labs in self._items:
            fname = os.path.basename(ipath)
            stem = os.path.splitext(fname)[0]
            if '@' in stem:
                group_key = stem.split('@')[0]
            else:
                group_key = stem
            if group_key not in group2id:
                group2id[group_key] = len(group2id)
            gid = group2id[group_key]
            self._group_ids.append(gid)

    def __len__(self):
        return len(self._items)
    
    def __getitem__(self, idx):
        p, labs = self._items[idx]
        group_id = self._group_ids[idx]
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            if self.strict:
                raise
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform is not None:
            img = self.transform(img)
        
        y = torch.zeros(self.num_classes, dtype=torch.float32)
        for name in labs:
            if name in self.cls2id:
                y[self.cls2id[name]] = 1.0
        
        # 20251121 return group_id
        if self.return_group_id:
            return img, y, group_id
        else:
            return img, y