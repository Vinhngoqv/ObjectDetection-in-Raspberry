import re
import cv2
import tensorflow as tf
import numpy as np

def load_labels(path='labels.txt'):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels

def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  # Get all output details
  scores = get_output_tensor(interpreter, 0)
  boxes = get_output_tensor(interpreter, 1)
  count = int(get_output_tensor(interpreter, 2))
  classes = get_output_tensor(interpreter, 3)
  # print(count)
  # print(scores)

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results

def main():
    labels = load_labels()
    interpreter = tf.lite.Interpreter(model_path = 'detect.tflite')
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    freq = cv2.getTickFrequency()

    colorB = {
      0: (0,255,0),
      1: (0,255,255),
      2: (128,128,0),
      3: (255,255,255),
      4: (128,128,128),
      5: (128,255,255)
    }

    cap = cv2.VideoCapture("Videotesting-1.mp4")          ### Change source video
    CAMERA_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    CAMERA_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  

    while cap.isOpened():
        t1 = cv2.getTickCount()
        ret, frame = cap.read()
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320,320))
        res = detect_objects(interpreter, img, 0.9)
        # print(res)

        for result in res:
            ymin, xmin, ymax, xmax = result['bounding_box']
            xmin = int(max(1,xmin * CAMERA_WIDTH))
            xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
            ymin = int(max(1, ymin * CAMERA_HEIGHT))
            ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
            
            colorCargo = colorB[int(result['class_id'])]
            cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),colorCargo,3)
            cv2.putText(frame,labels[int(result['class_id'])],(xmin+5, ymin+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,colorCargo,2,cv2.LINE_AA) 
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = int(1/time1)
        cv2.putText(frame,f"FPS: {frame_rate_calc}",(30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,0),2,cv2.LINE_AA)
        cv2.imshow('Cargo Clasification', frame)

        if cv2.waitKey(10) & 0xFF ==ord('q'):
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()