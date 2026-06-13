import tensorflow as tf
model = tf.keras.models.load_model('1.keras')
model.export('potato_model/1')
