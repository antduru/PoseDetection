def iou(bbox1, bbox2):
    x1 = max(bbox1['x1'], bbox2['x1'])
    y1 = max(bbox1['y1'], bbox2['y1'])
    x2 = min(bbox1['x2'], bbox2['x2'])
    y2 = min(bbox1['y2'], bbox2['y2'])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (bbox1['y2'] - bbox1['y1']) * (bbox1['x2'] - bbox1['x1'])
    a2 = (bbox2['y2'] - bbox2['y1']) * (bbox2['x2'] - bbox2['x1'])
    
    return i / (a1 + a2 - i)