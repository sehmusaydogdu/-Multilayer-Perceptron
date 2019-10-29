#define the safelog function
safelog <- function(x) {
  return (log(x + 1e-100))
}

#define the sigmoid function
sigmoid <- function(a) {
  return (1 / (1 + exp(-a)))
}

#read file data (images.csv and label.csv)
image_data_read <- read.csv("hw03_images.csv", header=FALSE, sep=",")
label_data_read <- read.csv("hw03_labels.csv", header=FALSE, sep=",")

#convert matrix (image_data_read and label_data_read) values
image_data_read <- as.matrix(image_data_read)
label_data_read <- as.matrix(label_data_read)

#image.csv file seperating test and training data
train_image <- image_data_read[0:500,]
test_image  <- image_data_read[500:999,]

#label.csv file seperating test and training data
train_label <- label_data_read[0:500,]
test_label  <- label_data_read[500:999,]



#read file data (initial_V.csv and initial_W)
V <- read.csv("initial_V.csv", header=FALSE, sep=",")
W <- read.csv("initial_W.csv", header=FALSE, sep=",")

#convert matrix (V and W) values
V <- as.matrix(V)
W <- as.matrix(W)


#N row size, K column size, dim(matrix(N X K)) 500x5
N <- length(train_label) 
K <- max(train_label)  

z <- sigmoid(cbind(1,train_image) %*% W)


#set learning parameters
H <- 20
eta <- 0.0005
epsilon <- 1e-3
max_iterasyon <- 500





