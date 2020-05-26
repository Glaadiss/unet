from model import *
from data import *
from sklearn.model_selection import train_test_split



# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

X, y, orig_X, orig_Y = get_data()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, random_state=2018)



model = unet((im_height, im_width, 1))
model_checkpoint = ModelCheckpoint('unet_berlin.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit(X_train, y_train,steps_per_epoch=300,epochs=1, callbacks=[model_checkpoint])

model.load_weights('unet_berlin.hdf5')
results = model.predict(X_valid)


saveResult("data/berlin/test", results, orig_X, orig_Y)