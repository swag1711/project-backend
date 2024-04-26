import tensorflow as tf
 
 
#load the saved model to predict
image_model = tf.keras.models.load_model("model/final_model.h5")
 
#export the model
__all__ = ['image_model']