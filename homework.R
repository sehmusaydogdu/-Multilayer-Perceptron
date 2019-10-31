# define the sigmoid function
sigmoid <- function(data , W) {
  result <- cbind(1, data) %*% W
  return (1 / (1 + exp(-result)))
}

#safelog function
safelog <- function(x) {
  return (log(x + 1e-100))
}

#softmax function
softmax <- function(Z,V) {
  result <- cbind(1, Z) %*% V
  scores <- exp(result - matrix(apply(result, MARGIN = 1, FUN = max), nrow = nrow(result), ncol = ncol(result), byrow = FALSE))
  scores <- scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE)
  return (scores)
}

#learning parameters
H <- 20
eta <- 0.0005
epsilon <- 1e-3
max_iteration <- 500

#read file data ('images.csv' and 'label.csv')
image_data_read_file <- read.csv("hw03_images.csv", header=FALSE, sep=",")
label_data_read_file <- read.csv("hw03_labels.csv", header=FALSE, sep=",")

#convert matrix ('image_data_read_file' and 'label_data_read_file') values
image_data_matrix <- as.matrix(image_data_read_file) # matrix [1000 X 784]
label_data_matrix <- as.matrix(label_data_read_file) # matrix [1000 X  1 ]


#read file data (initial_V.csv and initial_W)
initial_V_read_file <- read.csv("initial_V.csv", header=FALSE, sep=",")
initial_W_read_file <- read.csv("initial_W.csv", header=FALSE, sep=",")

#convert matrix ('initial_V_read_file' and 'initial_W_read_file') values
initial_V <- as.matrix(initial_V_read_file) # matrix [21  X 5 ]          
initial_W<- as.matrix(initial_W_read_file) # matrix [785 X 20]


# 'image_data_matrix' seperating test and training data
train_image <- image_data_matrix[0:500,]    #matrix [500 X 784]
test_image  <- image_data_matrix[500:999,]  #matrix [500 X 784]

# 'label_data_matrix' seperating test and training data
train_label <- label_data_matrix[0:500,]    #matrix [500 X 784]
test_label  <- label_data_matrix[500:999,]  #matrix [500 X 784]


N <- length(train_label)      # row size = 500
K <- max(train_label)         # class size = 5
D <- ncol(image_data_matrix)  # data features size = 784 (x1,x2,...x784)

# one-of-K-encoding
# Y_train_truth(matrix = [500X5] )
Y_train_truth <- matrix(0, N, K)
Y_train_truth[cbind(1:N, train_label)] <- 1

# one-of-K-encoding
# Y_test_truth(matrix = [500X5] )
Y_test_truth <- matrix(0, N, K)
Y_test_truth[cbind(1:N, test_label)] <- 1


# Hidden nodes calc (X*W)
Z <- sigmoid(train_image,initial_W)

# Output nodes calc (V*Z)
Y_train_predicted <- softmax(Z,initial_V)

objective_values <- -sum(Y_train_truth * safelog(Y_train_predicted))

# learn W and v using gradient descent and online learning
iteration <- 1

while (1) {

  y_transpose <- t(Y_train_truth - Y_train_predicted)
  image <- cbind(1, train_image)
  
  new_V <- eta * t(y_transpose %*% (cbind(1, Z)))
  new_W <- eta * t(image) %*% (t(initial_V[2:(H + 1),] %*% y_transpose) * Z[, 1:H] * (1 - Z[, 1:H]))
  
  initial_V <- initial_V + new_V
  initial_W <- initial_W + new_W
  
  # Hidden nodes calc (X*W)
  Z <- sigmoid(train_image, initial_W)
  
  # Output nodes calc (V*Z)
  Y_train_predicted <- softmax(Z,initial_V)
  
  objective_values <- c(objective_values, -sum(Y_train_truth * safelog(Y_train_predicted)))
  
  if (abs(objective_values[iteration + 1] - objective_values[iteration]) < epsilon | iteration >= max_iteration) {
    break
  }
  
  iteration <- iteration + 1
}

# plot objective function
plot(1:(iteration + 1), objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")


#Train Confusion matrix
y_train_confusion_matrix <- apply(Y_train_predicted, MARGIN = 1, FUN = which.max)
train_confusion_matrix <- table(y_train_confusion_matrix, train_label)
print(train_confusion_matrix)

#Test run
Z <- sigmoid(test_image,initial_W)
Y_test_predicted<- softmax(Z,initial_V)

#Test Confusion matrix
y_test_confusion_matrix<- apply(Y_test_predicted, MARGIN = 1, FUN = which.max)
test_confusion_matrix <- table(y_test_confusion_matrix, test_label)
print(test_confusion_matrix)


