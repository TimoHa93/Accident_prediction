import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

#initialization as in He et al. --> https://arxiv.org/abs/1502.01852
he_init = tf.contrib.layers.variance_scaling_initializer()

class DeepNeuralNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden_layers=3, n_neurons=50, optimizer_class=tf.train.AdamOptimizer, learning_rate=0.01, batch_size=30, activation=tf.nn.relu, initializer=he_init, batch_norm_momentum=None, random_state=None, max_checks_without_progress=20, show_progress=10, dropout_rate=None):
        #initializing the deep neural network with default hyperparameters
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.initializer = initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.random_state = random_state
        self.max_checks_without_progress = max_checks_without_progress
        self.show_progress = show_progress
        self.dropout_rate = dropout_rate
        self._session = None

    def _DeepNeuralNet(self, inputs): # members preceded by _ are private members
        for layer in range(self.n_hidden_layers):

            #if specified, handle dropout
            if self.dropout_rate:
                inputs = tf.layers.dropout(inputs, rate=self.dropout_rate, training=self._training)

            #hidden layer:
            inputs = tf.layers.dense(inputs, self.n_neurons, activation=self.activation, kernel_initializer=self.initializer, name = "hidden{}".format(layer+1))

            # if specified, apply batch normalization
            if self.batch_norm_momentum:
                inputs = tf.layers.batch_normalization(inputs, momentum=self.batch_norm_momentum, training=self._training)

            #activate the network
            inputs = self.activation(inputs, name="hidden{}_output".format(layer+1))
        return inputs

    def _tfComputationGraph(self, n_inputs, n_outputs):
        """Create the tf computation graph"""

        #if specified, apply random state
        if self.random_state:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        # Placeholders for training data, labels are class exclusive integers
        x = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.int32, shape=(None), name="y")

        # Create a training placeholder
        if self.batch_norm_momentum or self.dropout_rate:
            self._training = tf.placeholder_with_default(False, shape=[], name="training")
        else:
            self._training = None

        #output of hidden layers
        pre_output = self._DeepNeuralNet(x)

        #final output
        logits = tf.layers.dense(pre_output, n_outputs, kernel_initializer=he_init, name="logits")
        probabilities = tf.nn.softmax(logits, name="probabilities")

        #use TensorFlows reduce_mean() to compute the mean cross entropy
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

        #define the optimizer that will tweak the model parameters to minimize the cost function
        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            training_op = optimizer.minimize(loss)

        #evaluation of the model: accuracy
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        #create a node to initialize all variables and a Saver to save the trained model parameters
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self._X, self._y = x, y
        self._logits = logits
        self._probabilities = probabilities
        self._loss = loss
        self._training_op = training_op
        self._accuracy = accuracy
        self._init, self._saver = init, saver

    #create a function that closes the session
    def close_session(self):
        if self._session:
            self._session.close()

    def _get_model_parameters(self):
        with self._graph.as_default():
            all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {all_vars.op.name: value for var, value in zip(all_vars, self._session.run(all_vars))}


    #define a function that restores the value of all variables with TensorFlows assign ops
    def _restore_model_parameters(self, model_params):
        all_var_names = list(model_params.keys())

        #retrieve all the assignment ops in the graph
        assignment_ops = {all_var_name: self._graph.get_operation_by_name(all_var_name + "/Assign") for all_var_name in all_var_names}

        # fetch initialization
        init_vals = {all_var_name: assignment_op.inputs[1] for all_var_name, assignment_op in assignment_ops.items()}

        feed_dict = {init_vals[all_var_name]: model_params[all_var_name] for all_var_name in all_var_names}

        # Assign the trained value to all the variables in the graph
        self._session.run(assignment_ops, feed_dict=feed_dict)

    #define a function to train the model
    def fit(self, x, y, n_epochs=100, X_valid=None, y_valid=None):
        self.close_session()
        #get the number of features in x
        n_inputs = x.shape[1]

        # convert y labels to ints if necessary
        y = np.array(y)
        y_valid = np.array(y_valid)

        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        if len(y_valid.shape) == 2:
            y_valid = np.argmax(y_valid, axis=1)

        self.classes_ = np.unique(y) #get the classes in y
        n_outputs = len(self.classes_) #and get the number of distinct classes

        self.class_to_index_ = {label:index for index, label in enumerate(self.classes_)}
        labels = [self.class_to_index_[label] for label in y]
        y = np.array(labels, dtype=np.int32)

        self._graph = tf.Graph()

        #create the computation graph with self as default
        with self._graph.as_default():
            self._tfComputationGraph(n_inputs, n_outputs)

        # early stopping
        checks_without_progress = 0
        best_loss = np.float("inf")
        best_parameters = None

        self._session = tf.Session(graph=self._graph)

        #initialize all variables
        with self._session.as_default() as sess:
            self._init.run()
            num_training_instances = x.shape[0]
            for epoch in range(n_epochs):
                rnd_indices = np.random.permutation(num_training_instances)
                for rnd_index in np.array_split(rnd_indices, num_training_instances // self.batch_size):
                    x_batch, y_batch = x[rnd_index], y[rnd_index]
                    feed_dict = {self._X: x_batch, self._y: y_batch}
                    if self._training is not None:
                        feed_dict[self._training] = True
                    train_acc, _ = sess.run([self._accuracy, self._training_op], feed_dict)

                #implementing early stopping
                if X_valid is not None and y_valid is not None:
                    feed_dict_valid = {self._X: X_valid, self._y: y_valid}

                    val_acc, val_loss = sess.run([self._accuracy, self._loss], feed_dict=feed_dict_valid)

                #if specified, show training progress
                    if self.show_progress:
                        if epoch % self.show_progress == 0:
                            print("Epoch: {}, Current training accuracy: {:.3f}, Validation Accuracy: {:.3f}, Validation loss {:.4f}".format(epoch+1, train_acc, val_acc, val_loss))

                    #control for model improvement
                    if val_loss < best_loss:
                        best_loss = val_loss
                        checks_without_progress = 0
                        best_parameters = self._get_model_parameters()
                    else:
                        checks_without_progress += 1

                    if checks_without_progress > self.max_checks_without_progress:
                        print("Early stopping. Loss has not improved in {} epochs".format(self.max_checks_without_progress))
                        break

            #no validation
            else:
                if self.show_progress:
                    if epoch % self.show_progress == 0:
                        print("Epoch: {}, Current training accuracy: {:.3f}".format(epoch+1, train_acc))

        #if early stopping is specified, restore the best weight values
        if best_parameters:
            self._restore_model_parameters(best_parameters)
            return self

    #define a function that predicts the probabilities of each class in y
    def predict_probabilities(self, X):
        # Predict the probabilities of each class
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._probabilities.eval(feed_dict={self._X: X})

    #define a prediction function
    def predict(self, x):
        class_indices = np.argmax(self.predict_probabilities(x), axis=1)
        predictions = np.array([[self.classes_[class_idx]] for class_idx in class_indices], dtype=np.int32)
        return np.reshape(predictions, (-1,))

    #define a function to save the model
    def save(self, path):
        self._saver.save(self._session, path)
