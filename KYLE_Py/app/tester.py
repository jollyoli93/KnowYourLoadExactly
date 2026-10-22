from pathlib import Path

from app.ModelClasses.Load_Classify import Load_Classify, ModelCache
from PIL import Image
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "Models" / "crane_models" / "crane_classifier.onnx"
img_path = BASE_DIR / "images" / "concrete_skip.png"
print(img_path)
# sess = ModelCache.get_session()
classfiy = Load_Classify(DEFAULT_MODEL_PATH)
classfiy.load_model_onnx()
print(classfiy.predict_one_image(img_path))

# img_pil = Image.open(img_path).convert('RGB').resize((240,240))
# img = np.array(img_pil).astype('float32')
# img = img.transpose(2,0,1)
# img /= 255
# img = img[None, ...]

# input_name = sess.get_inputs()[0].name
# output_name = sess.get_outputs()[0].name

# input_cls = ['background',
#  'beams',
#  'concrete',
#  'formwork',
#  'pipes',
#  'plant',
#  'precast',
#  'rebar',
#  'skip']

# results = sess.run([output_name], {input_name: img})[0]
# print(input_cls[np.argmax(results)], results, results[0][np.argmax(results)])
