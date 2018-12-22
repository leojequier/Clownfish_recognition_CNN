data = read.table("Multi_utile.csv", header = T, sep = ";")
rm(other_info)

other_info = read.csv("utile.csv", header = T, sep = ";")


#add the scientificName, basisOfRecord and preparation column to data
i = 1
data[,4] = rep(NA, time(length(data[,1])))
for(i in seq(1, length(data[,1]))){
  data[i,4] = as.character(other_info$acceptedScientificName[other_info$gbifID == data[i,1]])
}
data[,4] = as.factor(data[,4])

for(i in seq(1, length(data[,1]))){
  data[i,5] = as.character(other_info$basisOfRecord[other_info$gbifID == data[i,1]])
}

for(i in seq(1, length(data[,1]))){
  data[i,6] = as.character(other_info$preparations[other_info$gbifID == data[i,1]])
}

for(i in seq(1, length(data[,1]))){
  data[i,7] = as.character(other_info$rightsHolder[other_info$gbifID == data[i,1]])
}
names(data)[4:7] = c("Species","BasisOfRecord", "Preparation", "rightHolder")

# merge the similar species together
levels(data$Species)
sort(summary(data$Species), decreasing = T)
which(is.na(data$Species))
which(data$Species == levels(data$Species)[16])
data$gbifID[1213]
which(data$Species == "Amphiprion_clarkii")
data$BasisOfRecord[1116]
summary(as.factor(data$BasisOfRecord))
data$Species[data$BasisOfRecord == "MachineObservation"]
data$identifier[data$BasisOfRecord == "HumanObservation"][1070:1100]
grep("inaturalist", data$identifier[data$BasisOfRecord == "HumanObservation"], invert = T, value = T )
data$identifier[data$BasisOfRecord == "Specimen"]
data$identifier[data$BasisOfRecord == "PreservedSpecimen"]
?grep()
#remove a useless specimen
data = data[-1454,]
data = data[-1213,]
grep("pictures.snsb", data$identifier[data$BasisOfRecord == "PreservedSpecimen"], invert = F)
data = data[!data$BasisOfRecord == "Specimen",]
data = data[!data$BasisOfRecord == "Specimen",]
data = data[data$BasisOfRecord %in% c("HumanObservation","MachineObservation" ),]
data$BasisOfRecord[1454]

levels(data$Species)[2] = paste(unlist(strsplit(levels(data$Species)[2], " "))[1:2], collapse = "_")
i = 1
for(i in seq(1,length(levels(data$Species)))){
  levels(data$Species)[i] = paste(unlist(strsplit(levels(data$Species)[i], " "))[1:2], collapse = "_")
}

write.table(data, "final_data.csv", sep = ";", row.names = F)
write.table(data[1:200,], "small_data.csv", sep = ";", row.names = F)

data = read.table("final_data.csv", sep = ";", header = T)

?write.table()
