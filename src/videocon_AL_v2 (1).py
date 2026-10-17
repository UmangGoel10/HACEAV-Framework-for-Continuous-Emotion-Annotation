import os
import pandas as pd
import numpy as np
import csv
import statistics
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential, clone_model
from keras.layers import Input, LSTM, Dense, Dropout
from keras.optimizers import Adam

keras.utils.set_random_seed(0)
rng = np.random.default_rng(0)

# =========================
# Column layout (one place, no magic numbers)
# =========================
target_features = ['Score', 'GSRmean_persen_diff', 'HRmean_persen_diff',
                   'valence_acc_video', 'arousal_acc_video',
                   'pw_HH', 'pw_HL', 'pw_LH', 'pw_LL',
                   'probe_HH', 'probe_HL', 'probe_LH', 'probe_LL']
FEATURE_IDX = slice(0, 9)   # model inputs
PW_IDX = slice(5, 9)        # pw_*    : label of the PREVIOUS window (an input)
LABEL_IDX = slice(9, 13)    # probe_* : label of the CURRENT window (what we predict)
assert target_features[PW_IDX] == ['pw_HH', 'pw_HL', 'pw_LH', 'pw_LL']
assert target_features[LABEL_IDX] == ['probe_HH', 'probe_HL', 'probe_LH', 'probe_LL']

# Fine-tuning settings used when the human answers a query
FINE_TUNE_LR = 1e-4
FINE_TUNE_EPOCHS = 3
REPLAY_SIZE = 64   # original training windows mixed in so the model does not forget them

# =========================
# Helper: Create sequences
# =========================
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back + 1):
        a = dataset[i:(i + look_back), FEATURE_IDX]
        dataX.append(a)
        label_vector = dataset[i + look_back - 1, LABEL_IDX]
        dataY.append(label_vector)
    return np.array(dataX), np.array(dataY)

# =========================
# Simulated human
# =========================
def ask_human(true_label):
    """Stand-in for a real person. The CSV already holds the true label of every
    window, so the 'human' simply returns it. Swap this function for a real
    question-and-answer step in a live system."""
    return true_label

# =========================
# Train initial model
# =========================
def model_train(training_data, look_back):
    trainX, trainY = create_dataset(training_data, look_back)
    trainX = trainX.astype('float32')
    trainY = trainY.astype('float32')

    n_timesteps, n_features = trainX.shape[1], trainX.shape[2]
    model = Sequential()
    model.add(Input(shape=(n_timesteps, n_features)))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=1, batch_size=8, verbose=1)
    return model

# =========================
# Update the existing model (no retraining from scratch)
# =========================
def fine_tune(model, labeled_X, labeled_Y, base_X, base_Y):
    """Continue training the current model on every example the human has
    labeled so far, plus a small random sample of the original training windows."""
    pick = rng.choice(len(base_X), size=min(REPLAY_SIZE, len(base_X)), replace=False)
    X = np.concatenate([np.array(labeled_X), base_X[pick]]).astype('float32')
    Y = np.concatenate([np.array(labeled_Y), base_Y[pick]]).astype('float32')
    model.fit(X, Y, epochs=FINE_TUNE_EPOCHS, batch_size=8, verbose=0)
    return model

# =========================
# Retrain + Track Performance
# =========================
def evaluate_and_retrain(model, user_data, T, base_windows, labeled_X, labeled_Y,
                         look_back, subject_id, video_id, retrain_csv_dir, plot_dir):
    """labeled_X / labeled_Y hold the human-labeled windows of this subject. They are
    extended IN PLACE, so they carry over from one video to the next."""
    probe_count = 0
    probe_steps, acc_list, loss_list = [], [], []

    for i in range(len(user_data) - look_back - 1):
        retrain_test_data = user_data[i:i + look_back]
        testX, testY = create_dataset(retrain_test_data, look_back)
        testX = testX.astype(np.float32)

        y_pred_probs = model.predict(testX, verbose=0)
        max_conf = np.max(y_pred_probs)
        predicted_class = np.argmax(y_pred_probs[0])

        if max_conf >= T:

            if (i + 1) < len(user_data):
                one_hot_label = np.zeros(4, dtype=int)
                one_hot_label[predicted_class] = True
                user_data[i + 1, PW_IDX] = one_hot_label
        else:
            probe_count += 1
            # Ask the human, remember the answer, and update the existing model
            answer = ask_human(testY[0])
            labeled_X.append(testX[0])
            labeled_Y.append(answer)
            model = fine_tune(model, labeled_X, labeled_Y, *base_windows)

            # Evaluate after this probe step
            evalX, evalY = create_dataset(user_data, look_back)
            evalX = evalX.astype(np.float32)
            evalY = evalY.astype(np.float32)
            eval_loss, eval_acc = model.evaluate(evalX, evalY, verbose=0)

            probe_steps.append(probe_count)
            acc_list.append(eval_acc)
            loss_list.append(eval_loss)

    # Save retraining CSV for this subject/video
    os.makedirs(retrain_csv_dir, exist_ok=True)
    csv_path = os.path.join(retrain_csv_dir, f"subject_{subject_id}_video_{video_id}_retrain_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Probe", "Accuracy", "Loss"])
        for p, a, l in zip(probe_steps, acc_list, loss_list):
            writer.writerow([p, a, l])

    # Plot
    os.makedirs(plot_dir, exist_ok=True)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(probe_steps, acc_list, 'g-', label='Accuracy')
    ax2.plot(probe_steps, loss_list, 'b-', label='Loss')

    ax1.set_xlabel('Probes')
    ax1.set_ylabel('Accuracy', color='g')
    ax1.set_ylim(0.7, 1.0)
    ax2.set_ylabel('Loss', color='b')

    fig.suptitle(f"Subject {subject_id} - Video {video_id}")
    plt.savefig(os.path.join(plot_dir, f"subject_{subject_id}_video_{video_id}_plot.png"))
    plt.close()

    return probe_count, model

# =========================
# Load and prepare data
# =========================
df = pd.read_csv("updated_modified_alluser_norm_mean_normalized_allfeatures.csv")

# Check what Probe and prev_window mean: prev_window must be the previous row's Probe
# inside each participant-video. If this fails, the target columns need rethinking.
prev_probe = df.groupby(['P_id', 'video_flag'])['Probe'].shift(1)
has_prev = prev_probe.notna()
assert (df.loc[has_prev, 'prev_window'] == prev_probe[has_prev]).all(), \
    "prev_window is not the previous window's Probe; check the target columns"

data = df[['Score','GSRmean_persen_diff','HRmean_persen_diff','valence_acc_video',
           'arousal_acc_video','prev_window','P_id','Probe','video_flag']]

pw_onehot = pd.get_dummies(data['prev_window'], prefix='pw')
probe_onehot = pd.get_dummies(data['Probe'], prefix='probe')
data = pd.concat([data.drop(['prev_window', 'Probe'], axis=1), pw_onehot, probe_onehot], axis=1)

subject_ids = list(range(1, 43))
Number_of_training_videos = [1, 2]
Number_of_retraining_videos = [3, 4, 5, 6]
Number_of_test_videos = [7, 8]

training_dataset = pd.DataFrame()
retraining_dataset = pd.DataFrame()
test_dataset = pd.DataFrame()

for s in subject_ids:
    for v in Number_of_training_videos:
        training_dataset = pd.concat([training_dataset, data[(data['P_id'] == s) & (data['video_flag'] == v)]])
    for v in Number_of_retraining_videos:
        retraining_dataset = pd.concat([retraining_dataset, data[(data['P_id'] == s) & (data['video_flag'] == v)]])
    for v in Number_of_test_videos:
        test_dataset = pd.concat([test_dataset, data[(data['P_id'] == s) & (data['video_flag'] == v)]])

# =========================
# Main loop
# =========================
T = 0.80
look_back = 3
training_data = training_dataset[target_features].to_numpy(dtype='float32')
lstm_model = model_train(training_data, look_back)
base_windows = create_dataset(training_data, look_back)

summary_rows = []
for s in subject_ids:
    model_copy = clone_model(lstm_model)
    model_copy.set_weights(lstm_model.get_weights())
    model_copy.compile(optimizer=Adam(learning_rate=FINE_TUNE_LR),
                       loss='categorical_crossentropy', metrics=['accuracy'])

    # Per-subject memory: lives across videos 3-6, starts empty for each new subject
    labeled_X, labeled_Y = [], []

    for v in Number_of_retraining_videos:
        retraining_user_data = retraining_dataset[(retraining_dataset['P_id'] == s) &
                                                  (retraining_dataset['video_flag'] == v)][target_features].to_numpy(dtype='float32')
        probe_count, model_copy = evaluate_and_retrain(
            model_copy,
            retraining_user_data,
            T,
            base_windows,
            labeled_X,
            labeled_Y,
            look_back,
            subject_id=s,
            video_id=v,
            retrain_csv_dir="retrain_csvs",
            plot_dir="retrain_plots"
        )
        n_windows = len(retraining_user_data) - look_back - 1
        summary_rows.append((s, v, n_windows, probe_count))
        print(f"subject {s} video {v}: asked the human {probe_count} of {n_windows} windows "
              f"(labeled examples kept so far: {len(labeled_X)})")

with open(os.path.join("retrain_csvs", "human_queries_summary.csv"), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Subject", "Video", "Windows", "HumanQueries"])
    writer.writerows(summary_rows)

total_q = sum(r[3] for r in summary_rows)
total_w = sum(r[2] for r in summary_rows)
print(f"Retraining with plots and CSV logging complete. Human asked {total_q} of {total_w} windows ({total_q / total_w:.1%}).")
