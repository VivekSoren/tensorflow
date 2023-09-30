import argparse
import numpy as np
import cv2
import os
import tensorflow as tf
import subprocess
from label_image import load_labels, read_tensor_from_image_file
tf.compat.v1.disable_eager_execution()

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def read_from_video_file(file_name):
  video_path = file_name
  cap = cv2.VideoCapture(video_path)

  if not cap.isOpened():
    raise ValueError("Could not open the file")
  
  frame_dir = "tensorflow/examples/label_image/data/frames"
  os.makedirs(frame_dir, exist_ok=True)
 
  frame_count = 0
  while True:
    ret, frame = cap.read()
    if not ret:
      break

    frame_filename = os.path.join(frame_dir, f'fname_{frame_count:04d}.png')
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

  cap.release()
  print("Video read")
  return frame_count, frame_dir

def get_predictions_from_video(graph, frame_dir, frame_count):
  
  results_list = []

  for frame_num in range(frame_count):
    frame_filename = os.path.join(frame_dir, f'fname_{frame_num:04d}.png')
    t = read_tensor_from_image_file(frame_filename, 
                                    input_height=input_height, 
                                    input_width=input_width, 
                                    input_mean=input_mean, 
                                    input_std=input_std)
    
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.compat.v1.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
      })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k:
      results_list.append({labels[i], results[i]})
      break
  return results_list

def generate_pred_video(frame_dir, pred_list):

  threshold:float = 0.8
  
  # Create a video writer
  output_path = "tensorflow/examples/label_image/output.mp4"
  frame_width, frame_height = 640, 480
  fps = 1
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

  for i, label in enumerate(pred_list):
    frame_filename = os.path.join(frame_dir, f'fname_{i:04d}.png')
    frame = cv2.imread(frame_filename)
    ele1, ele2 = label
    if type(ele1) == str:
      pred = ele1
      score = ele2
    else:
      score = ele1
      pred = ele2

    # Annotate the frame iwth label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    if score > threshold:
      font_color = (0, 255, 0)    # Green
    else:
      font_color = (0, 0, 255)    # Red
    org = (10, frame_height-10)
    label = " ".join((("{:.2f}%".format(score*100)), pred))
    frame = cv2.putText(frame, label, org, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # Write annotated frame to output video
    out.write(frame)

    # Remove frames
    os.remove(frame_filename)
  
  out.release()
  os.rmdir(frame_dir)
  print("Prediction Video Created")
  return output_path


if __name__ == "__main__":
  file_name = "tensorflow/examples/label_image/data/input.mp4"
  model_file = \
    "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
  label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "input"
  output_layer = "InceptionV3/Predictions/Reshape_1"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)
  frame_count, frame_dir = read_from_video_file(file_name)
  preds_list = get_predictions_from_video(graph, frame_dir, frame_count)
  output_path = generate_pred_video(frame_dir, preds_list)

  try:
    subprocess.Popen(['start', '', output_path], shell=True)  # Windows
  except FileNotFoundError:
      try:
          subprocess.Popen(['open', output_path])  # macOS
      except FileNotFoundError:
          subprocess.Popen(['xdg-open', output_path])  # Linux