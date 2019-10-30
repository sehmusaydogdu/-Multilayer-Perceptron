safelog <- function(x) {
  return (log(x + 1e-100))
}

# read data into memory
image_data_set <- as.matrix(read.csv("hw03_images.csv", header=FALSE, sep=","))
label_data_set <- as.matrix(read.csv("hw03_labels.csv", header=FALSE, sep=","))

#split train and test data_sets
train_image_data_set <- image_data_set[0:500,]
test_image_data_set <- image_data_set[500:999,]

train_label_data_set <- label_data_set[0:500,]
test_label_data_set <-label_data_set[500:999,]

# create truth vector
Y_truth_train <- train_label_data_set
Y_truth_test <- test_label_data_set

# get number of classes and number of samples
K_train <- max(label_data_set)
N_train <- length(Y_truth_train)

K_test <- max(label_data_set)
N_test <- length(Y_truth_test)

# get number of samples and number of features
N <- length(train_label_data_set)
D <- ncol(train_image_data_set)

# one-of-K-encoding
y_truth_train <- matrix(0, N_train, K_train)
y_truth_train[cbind(1:N_train, Y_truth_train)] <- 1

y_truth_test <- matrix(0, N_test, K_test)
y_truth_test[cbind(1:N_test, Y_truth_test)] <- 1

# define the sigmoid function
sigmoid <- function(a) {
  return (1 / (1 + exp(-a)))
}

# define the softmax function
softmax <- function(M) {
  scores <- exp(M - matrix(apply(M, MARGIN = 1, FUN = max), nrow = nrow(M), ncol = ncol(M), byrow = FALSE))
  scores <- scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE)
  return (scores)
}

# set learning parameters
eta <- 0.0005
epsilon <- 1e-3
H <- 20
max_iteration <- 500

# initalize W and v
W <- as.matrix(read.csv("initial_W.csv", header=FALSE, sep=","))
v <- as.matrix(read.csv("initial_V.csv", header=FALSE, sep=","))

Z <- sigmoid(cbind(1, train_image_data_set) %*% W)
y_predicted_train <- softmax(cbind(1,Z) %*% v)
objective_values <- -sum(y_truth_train * safelog(y_predicted_train))

# learn W and v using gradient descent and batch learning
iteration <- 1
while (1) {
 
  # calculate hidden nodes
  Z <- sigmoid(cbind(1, train_image_data_set) %*% W)
  
  # calculate output nodes
  y_predicted_train <- softmax(cbind(1, Z) %*% v)
  
  # calculate delta
  delta_v <- eta * t(t(y_truth_train - y_predicted_train) %*% (cbind(1, Z)))
  delta_W <- eta * t(cbind(1, train_image_data_set)) %*% (t(v[2:(H + 1),] %*% t(y_truth_train- y_predicted_train)) * Z[, 1:H] * (1 - Z[, 1:H]))
  
  v <- v + delta_v
  W <- W + delta_W
  
  Z <- sigmoid(cbind(1, train_image_data_set) %*% W)
  y_predicted_train <- softmax(cbind(1, Z) %*% v)

  objective_values <- c(objective_values, -sum(y_truth_train * safelog(y_predicted_train)))
  
  if (abs(objective_values[iteration + 1] - objective_values[iteration]) < epsilon | iteration >= max_iteration) {
    break
  }
  
  iteration <- iteration + 1
}

# plot objective function during iterations
plot(1:(iteration + 1), objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

# run train classification
Z_train <- sigmoid(cbind(1, train_image_data_set) %*% W)
y_predicted_train <- softmax(cbind(1, Z_train) %*% v)

# calculate train confusion matrix
y_predicted_confusion_train <- apply(y_predicted_train, MARGIN = 1, FUN = which.max)
confusion_matrix_train <- table(y_predicted_confusion_train, Y_truth_train)
print(confusion_matrix_train)

# run test classification
Z_test <- sigmoid(cbind(1, test_image_data_set) %*% W)
y_predicted_test<- softmax(cbind(1, Z_test) %*% v)

# calculate test confusion matrix
y_predicted_confusion_test<- apply(y_predicted_test, MARGIN = 1, FUN = which.max)
confusion_matrix_test <- table(y_predicted_confusion_test, Y_truth_test)
print(confusion_matrix_test)
