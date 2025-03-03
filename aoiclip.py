from flask import Flask, request, jsonify, send_file
import cv2
import os
import shutil
import tempfile
from autodistill_clip import CLIP
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from autodistill.core.composed_detection_model import ComposedDetectionModel
import supervision as sv
import torch
from PIL import Image
import datetime
from torchvision.ops import box_convert
app = Flask(__name__)
outputfolderName='output'
os.makedirs(outputfolderName,exist_ok=True)

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    classes = request.form.getlist('class')[0].split(',')  # List of class names
    prompt = request.form.getlist('prompt')  # Default prompt
    threshold = float(request.form.get('threshold', 0.5))  # Default threshold
    
    # Create a temporary directory to store frames
    frame_dir = tempfile.mkdtemp()
    
    # Save video temporarily
    temp_video_path = os.path.join(tempfile.gettempdir(), video_file.filename)
    video_file.save(temp_video_path)
    
    # Initialize model
    SAMCLIP = ComposedDetectionModel(
        detection_model=GroundedSAM(
            CaptionOntology({ k : k for k in prompt[0].split(',')}),box_threshold=threshold
        ),
        classification_model=CLIP(
            CaptionOntology({ k : k for k in classes})
        )
    )
    
    cap = cv2.VideoCapture(temp_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = os.path.join(outputfolderName, 'output_'+video_file.filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = os.path.join(frame_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_path, frame)  # Save frame
        # frame = Image.open(frame_path)
        results = SAMCLIP.predict(frame_path)
        # print(results)
        # print('-----------------------')
        labels = [
            f"{classes[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _ ,_ in results
                ]
        boxes_len=len(results.xyxy)
        if boxes_len>0:
            annotated_frame = annotator.annotate(scene=frame.copy(), detections=results)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, labels=labels, detections=results)
            out.write(annotated_frame)
            h, w, _ = frame.shape
            for box_number in range(boxes_len):
                # boxes =  torch.from_numpy(results.xyxy[box_number]) * torch.Tensor([w, h, w, h])
                # boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
                boxes=int(results.xyxy[box_number][0]),int(results.xyxy[box_number][1]),int(results.xyxy[box_number][2]),int(results.xyxy[box_number][3])
                x1, y1, x2, y2 = int(boxes[0] ), int(boxes[1] ), int(boxes[2] ), int(boxes[3] )
                box_width = x2 - x1
                box_height = y2 - y1
                frame_info={
                "Processed frame": frame_count,
                "timestamp":str(datetime.timedelta(seconds=(frame_count/fps))),
                'milisecond_start':int((frame_count/fps)*1000),
                'milisecond_end':int(((frame_count+1)/fps)*1000),
                'ojbect_number':box_number,    
                'x':x1,
                'y':y1,
                'w':box_width,
                'h':box_height}
                print(frame_info)

        else:
            out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    os.remove(temp_video_path)  # Clean up input file
    shutil.rmtree(frame_dir)  # Delete the folder containing frames
    
    return send_file(output_path, as_attachment=True, download_name='output.mp4')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
