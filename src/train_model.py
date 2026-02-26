# src/train_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import the load_data function
from data_preprocessing import load_data

# ------------------------------
# 0️⃣ Load Dataset
# ------------------------------
X_train, X_val, y_train, y_val, X_test, y_test = load_data()

# ------------------------------
# 1️⃣ Build CNN Model
# ------------------------------
def build_cnn(input_shape=(48,48,1), num_classes=7):
    model = Sequential()

    # Conv Block 1
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Conv Block 2
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Conv Block 3
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Fully connected
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# ------------------------------
# 2️⃣ Compile Model
# ------------------------------
model = build_cnn()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------------
# 3️⃣ Callbacks
# ------------------------------
checkpoint = ModelCheckpoint("models/fer_cnn.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ------------------------------
# 4️⃣ Train Model
# ------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[checkpoint, early_stop]
)

# ------------------------------
# 5️⃣ Evaluate on Test Set
# ------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")