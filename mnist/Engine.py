import Model
import DataSet
import tensorflow as tf
import os

MODEL_DIR = "networks"

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

model = Model.Model()

summary = tf.merge_all_summaries()

session = tf.Session()
session.run(tf.initialize_all_variables())

writer = tf.train.SummaryWriter('summary-log', session.graph_def)

saver = tf.train.Saver(max_to_keep=1000000)
checkpoint = tf.train.get_checkpoint_state(MODEL_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(session, checkpoint.model_checkpoint_path)
    print("Loaded checkpoint: {}".format(checkpoint.model_checkpoint_path))
else:
    print("Unable to load checkpoint")

counter = 0

print(len(DataSet.TRAIN_DATASET.images))

saver.save(session, os.path.join(MODEL_DIR, "network"), global_step=counter)

for epoch in range(3):
    print(epoch)
    for images, labels in DataSet.iter_batches(50):
        counter += 1
        if counter % 100 == 0:
            print(counter)

            acc, summ = session.run([model.accuracy, summary], feed_dict = {
                model.input_var: images,
                model.corr_labels: labels,
                model.keep_prob: 1.0
            })

            writer.add_summary(summ, counter)
            print("iteration {}, training accuracy {}".format(counter, acc))

        session.run([model.train], feed_dict = {
            model.input_var: images,
            model.corr_labels: labels,
            model.keep_prob: 0.5
        })

    writer.flush()

    saver.save(session, os.path.join(MODEL_DIR, "network"), global_step=counter)

    acc = model.accuracy.eval(feed_dict = {
        model.input_var: DataSet.TEST_DATASET.images[:500],
        model.corr_labels: DataSet.TEST_DATASET.labels[:500],
        model.keep_prob: 1.0
    })
    print("iteration {}, test accuracy {}".format(counter, acc))


corr = 0
for images, labels in DataSet.iter_batches(500, DataSet.TEST_DATASET):
    corr += sum(model.correct.eval(feed_dict = {
        model.input_var: images,
        model.corr_labels: labels,
        model.keep_prob: 1.0
    }))
tot = len(DataSet.TEST_DATASET.images)
print("{}/{} = {}".format(corr, tot, float(corr) / tot))
