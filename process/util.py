import math

def iou(bbox1, bbox2):
    x1 = max(bbox1['x1'], bbox2['x1'])
    y1 = max(bbox1['y1'], bbox2['y1'])
    x2 = min(bbox1['x2'], bbox2['x2'])
    y2 = min(bbox1['y2'], bbox2['y2'])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (bbox1['y2'] - bbox1['y1']) * (bbox1['x2'] - bbox1['x1'])
    a2 = (bbox2['y2'] - bbox2['y1']) * (bbox2['x2'] - bbox2['x1'])
    
    return i / (a1 + a2 - i)

def diagonal_id(bbox1, bbox2):
    '''
        1 | 2 | 3
        4 |   | 6
        7 | 8 | 9

        finds where bbox2 lies. First finds it like below, 
        then mappes to the above representation.

        3 | 2 | 1
        4 |   | 0
        5 | 6 | 7
    '''
    ox1 = (bbox1['x1'] + bbox1['x2']) / 2
    oy1 = (bbox1['y1'] + bbox1['y2']) / 2
    ox2 = (bbox2['x1'] + bbox2['x2']) / 2
    oy2 = (bbox2['y1'] + bbox2['y2']) / 2
    
    vx, vy = (ox2 - ox1), (oy2 - oy1) # vector goes from bb1 to bb2

    omega = math.atan2(vy, vx) # find the angle between x axis

    degree = (int(180 * omega / math.pi) + 360 + 30) % 360
    index = int(8 * degree / 360)
    
    mapper = [6, 3, 2, 1, 4, 7, 8, 9]
    return mapper[index] # mapps the result here

def get_joint(joint_id, joint_list):
    for joint in joint_list:
        if joint['id'] == joint_id: return joint
    return None

def dist(person_object, from_k, to_k, joint_dict, epsilon=0.001):
    from_joint = get_joint(joint_dict[from_k], person_object['joints'])
    to_joint = get_joint(joint_dict[to_k], person_object['joints'])

    if from_joint and to_joint:
        d_x, d_y = (to_joint['x'] - from_joint['x']), (to_joint['y'] - from_joint['y'])
        length = math.sqrt(d_x**2 + d_y**2)
        if abs(length) < epsilon:
            return (d_x / epsilon), (d_y / epsilon)
        return (d_x / length), (d_y / length)

    return 0, 0