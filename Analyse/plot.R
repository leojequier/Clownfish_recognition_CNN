
#read validation accuracy dataset
dat = read.table("accuracies.csv", header = F, sep = ",")
dat = as.matrix(dat)
rownames(dat) = c()
colnames(dat) = c()
dat = t(dat)
head = dat[1,]
dat = dat[2:(nrow(dat)-1),]
dat = as.numeric(dat)
dat = matrix(dat, ncol = 12)
df =  as.data.frame(dat)
names(df) = head

T4f1 = read.table("accuracies_TL4f1.csv", header = F, sep = ",")
T4f1 = as.matrix(T4f1)
rownames(T4f1) = c()
colnames(T4f1) = c()
T4f1 = t(T4f1)
head = T4f1[1,]
T4f1 = T4f1[2:(nrow(T4f1)-1),]
T4f1 = as.numeric(T4f1)
T4f1 = matrix(T4f1, ncol = 2)
T4f1 =  as.data.frame(T4f1)
names(T4f1) = head


T2f1 = read.table("accuracies_TL2f1.csv", header = F, sep = ",")
T2f1 = as.matrix(T2f1)
rownames(T2f1) = c()
colnames(T2f1) = c()
T2f1 = t(T2f1)
head = T2f1[1,]
T2f1 = T2f1[2:(nrow(T2f1)-1),]
T2f1 = as.numeric(T2f1)
T2f1 = matrix(T2f1, ncol = 2)
T2f1 =  as.data.frame(T2f1)
names(T2f1) = head
#--------------------------------------------------------------------------------------------------
#read training accuracy datasets:

traindat = read.table("train_accuracies.csv", header = F, sep = ",")
traindat = as.matrix(traindat)
rownames(traindat) = c()
colnames(traindat) = c()
traindat = t(traindat)
head = traindat[1,]
traindat = traindat[2:(nrow(traindat)-1),]
traindat = as.numeric(traindat)
traindat = matrix(traindat, ncol = 6)
traindat =  as.data.frame(traindat)
names(traindat) = head

trainT4f1 = read.table("train_accuracy_TL4f1.csv", header = F, sep = ",")
trainT4f1 = as.matrix(trainT4f1)
rownames(trainT4f1) = c()
colnames(trainT4f1) = c()
trainT4f1 = t(trainT4f1)
head = trainT4f1[1,]
trainT4f1 = trainT4f1[2:(nrow(trainT4f1)-1),]
trainT4f1 = as.numeric(trainT4f1)
trainT4f1 = matrix(trainT4f1, ncol = 2)
trainT4f1 =  as.data.frame(trainT4f1)
names(trainT4f1) = head


trainT2f1 = read.table("train_accuracy_TL2f1.csv", header = F, sep = ",")
trainT2f1 = as.matrix(trainT2f1)
rownames(trainT2f1) = c()
colnames(trainT2f1) = c()
trainT2f1 = t(trainT2f1)
head = trainT2f1[1,]
trainT2f1 = trainT2f1[2:(nrow(trainT2f1)-1),]
trainT2f1 = as.numeric(trainT2f1)
trainT2f1 = matrix(trainT2f1, ncol = 2)
trainT2f1 =  as.data.frame(trainT2f1)
names(trainT2f1) = head









#accuracy 2 families

plot(c(1:length(df$acc_R2f1), 1:length(T2f1$acc_T2f1)), c(traindat$acc_R2f1, trainT2f1$acc_TL2f1), type = "n",
     main = "", xlab = "Epoch", ylab = "Accuracy")
points(1:length(df$acc_R2f1),df$acc_R2f1, type = "l", col = "blue")
points(1:length(T2f1$acc_T2f1), T2f1$acc_T2f1, type  ="l", col = "red")
points(1:length(traindat$acc_R2f1),traindat$acc_R2f1, type = "l",lty = 2, col = "blue")
points(1:length(trainT2f1$acc_TL2f1), trainT2f1$acc_TL2f1, type  ="l",lty = 2, col = "red")



#loss 2 families

plot(c(1:length(df$loss_R2f1), 1:length(T2f1$loss_T2f1)), c(traindat$loss_R2f1, trainT2f1$loss_TL2f1), type = "n",
     main = "", xlab = "Epoch", ylab = "Loss")
points(1:length(df$loss_R2f1),df$loss_R2f1, type = "l", col = "blue")
points(1:length(T2f1$loss_T2f1), T2f1$loss_T2f1, type  ="l", col = "red")
points(1:length(traindat$loss_R2f1),traindat$loss_R2f1, type = "l",lty = 2, col = "blue")
points(1:length(trainT2f1$loss_TL2f1), trainT2f1$loss_TL2f1, type  ="l",lty = 2, col = "red")

#legend 
plot.new()
par(xpd=TRUE)
legend(x = "center", legend = c("RI", "TL", "Training", "Validation"), fill = c("red", "blue", NA, NA), lty = c(NA, NA, 1,2))
par(xpd = FALSE)

#accuracy 4 families
plot(c(1:length(traindat$acc_R4f1), 1:length(trainT4f1$acc_TL4f1)), c(traindat$acc_R4f1, trainT4f1$acc_TL4f1), type = "n",
     main = "", xlab = "Epoch", ylab = "Accuracy")
points(1:length(df$acc_R4f1),df$acc_R4f1, type = "l", col = "blue")
points(1:length(T4f1$acc_T4f1), T4f1$acc_T4f1, type  ="l", col = "red")
points(1:length(traindat$acc_R4f1),traindat$acc_R4f1, type = "l",lty = 2, col = "blue")
points(1:length(trainT4f1$acc_TL4f1), trainT4f1$acc_TL4f1, type  ="l",lty = 2, col = "red")



#loss 4 families
plot(c(1:length(traindat$loss_R2f1), 1:length(trainT4f1$loss_TL4f1)), c(df$loss_R2f1, trainT4f1$loss_TL4f1), type = "n",
     main = "", xlab = "Epoch", ylab = "Loss")

points(1:length(df$loss_R2f1),df$loss_R2f1, type = "l", col = "blue")
points(1:length(T4f1$loss_T4f1), T4f1$loss_T4f1, type  ="l", col = "red")
points(1:length(traindat$loss_R2f1),traindat$loss_R2f1, type = "l",lty = 2, col = "blue")
points(1:length(trainT4f1$loss_TL4f1), trainT4f1$loss_TL4f1, type  ="l", lty = 2, col = "red")


#accuracy 4 families with other
plot(1:length(traindat$acc_TL4f2), traindat$acc_TL4f2, type = "n",
     main = "", xlab = "Epoch", ylab = "Accuracy")
points(1:length(df$acc_T4f2), df$acc_T4f2, type = "l", col = "blue")
points(1:length(traindat$acc_TL4f2),traindat$acc_TL4f2, type = "l",lty = 2, col = "blue")



#loss 4 families with other
plot(c(1:length(traindat$loss_TL4f2),1:length(df$loss_T4f2)), c(traindat$loss_TL4f2,df$loss_T2f2), type = "n",
     main = "", xlab = "Epoch", ylab = "Loss")

points(1:length(df$loss_T4f2), df$loss_T2f2, type = "l", col = "blue")
points(1:length(traindat$loss_TL4f2),traindat$loss_TL4f2, type = "l",lty = 2, col = "blue")

#legend
plot.new()
par(xpd=TRUE)
legend(x = "center", legend = c("Training", "Validation"),  lty = c(1,2))
par(xpd = FALSE)
