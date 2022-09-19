import tensorflow as tf
import cv2
import numpy as np

model = tf.saved_model.load('./dataset/saved_models/saved_model_0.55200845')
infer = model.signatures["serving_default"]

img = cv2.imread('./dataset/images/White_2022-08-18-09-52-33_TRITON_LEFT_RGB.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (550, 550)).astype(np.float32)
output = infer(tf.constant(img[None, ...]))



_h = img.shape[0]
_w = img.shape[1]

det_num = output['num_detections'][0].numpy()
det_boxes = output['detection_boxes'][0][:det_num]
det_boxes = det_boxes.numpy()*np.array([_h,_w,_h,_w])
det_masks = output['detection_masks'][0][:det_num].numpy()

det_scores = output['detection_scores'][0][:det_num].numpy()
det_classes = output['detection_classes'][0][:det_num].numpy()

for i in range(det_num):
    score = det_scores[i]
    if score > 0.5:
        box = det_boxes[i].astype(int)
        _class = det_classes[i]
        cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
        print(str(box[1] / 550.0) + ' ' + str(box[0] / 550.0) + ' ' + str(box[3] / 550.0) + ' ' + str(box[2] / 550.0))
        cv2.putText(img, str(_class)+'; '+str(round(score,2)), (box[1], box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), lineType=cv2.LINE_AA)
        mask = det_masks[i]
        print(mask.shape)
        mask = cv2.resize(mask, (_w, _h))
        print(_w)
        mask = (mask > 0.5)
        roi = img[mask]
        blended = roi.astype("uint8")
        img[mask] = blended*[0,0,1]

cv2.imwrite("out.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


