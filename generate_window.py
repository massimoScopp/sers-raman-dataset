import seglearn

def generate(window_size=1, overlap=0, shuffle=False):
    s = seglearn.SegmentX(width=window_size, overlap=overlap, shuffle=shuffle)
    return s
