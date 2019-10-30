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
  scores <- cbind(1, Z) %*% V
  scores <- exp(scores - matrix(apply(scores, MARGIN = 1, FUN = max), nrow = nrow(scores), ncol = ncol(scores), byrow = FALSE))
  scores <- scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE)
  return (scores)
}

#learning parameters
H <- 20
eta <- 0.0005
epsilon <- 1e-3
max_iterasyon <- 500

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
initial_W <- as.matrix(initial_W_read_file) # matrix [785 X 20]


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
Y_truth <- matrix(0, N, K)
Y_truth[cbind(1:N, Y_truth)] <- 1

# Hidden nodes calc (X*W)
Z <- sigmoid(train_image,initial_W)

# Output nodes calc (V*Z)
Y_predicted <- softmax(Z,initial_V)

objective_values <- -sum(Y_truth * safelog(Y_predicted))

iteration <- 1




