from mmocr.utils.ocr import MMOCR
import os

# Load models into memory
ocr = MMOCR(det='TextSnake', recog=None)

# Inference
root = 'test_DD'
out = 'out'
for root,_,files in os.walk(root):
            for file in files:
                results = ocr.readtext(os.path.join(root,file),  output=os.path.join(out,file), print_result=True, imshow=False)