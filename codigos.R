# Clasificador de bayes ingenuo

## empezamos instalando paquetes
install.packages("e1071")
library("e1071")

install.packages("klaR")
library("klaR")

install.packages("caret")
library("caret")

## lectura de datos

datos <- read.csv("/home/agabad/Downloads/germancredit.csv")

head(datos)
summary(datos)

datos$Default <- as.factor(datos$Default)
datos$installment <- as.factor(datos$installment)
datos$residence <- as.factor(datos$residence)
datos$cards <- as.factor(datos$cards)
datos$liable <- as.factor(datos$liable)

summary(datos)

## verificamos existencia de datos faltantes
sum(is.na(datos))


## particionamos los datos (TRAIN y TEST)
particion <- createDataPartition(datos$Default,
                                 p=.8,
                                 list=FALSE,
                                 times=1)

particion

TRAIN <- datos[particion,]
TEST <- datos[-particion,]

## ajustamos un modelo de naive bayes

modelo <- train(x=subset(TRAIN, select=-Default),
                y=TRAIN$Default,
                method="nb")

modelo
modelo$finalModel

prediccion <- predict(modelo, TEST)

confusionMatrix(prediccion, TEST$Default)

## calibrando los modelos
umbral <- 0.1

prediccion.clases <- as.factor(ifelse(predict(modelo, TEST, type="prob")[,2]>umbral, 1, 0))

matriz <- confusionMatrix(prediccion.clases, TEST$Default)

## encontrando umbral optimo
umbral <- seq(0,1,.001)

errores<-NULL
sensitivity<-NULL
for (i in 1:length(umbral))
{prediccion.clases <- as.factor(ifelse(predict(modelo, TEST, type="prob")[,2]>umbral[i], 1, 0))
  
  matriz <- confusionMatrix(prediccion.clases, TEST$Default)
  errores[i]<-1-matriz$overall[1]
  sensitivity[i] <- matriz$byClass[1]
print(i/length(umbral))  
}

plot(y=errores,x=umbral,type="b")

plot(y=sensitivity,x=umbral,type="b")


## graficamos curva ROC
datos.roc <- datos
datos.roc$Default <- as.factor(ifelse(datos.roc$Default=="1","Si","No"))

modelo<-suppressWarnings(train(x=subset(datos.roc,select=-Default),
                               y=datos.roc$Default,
                               method="nb",
                               trControl=trainControl(method='cv',
                                                      number=5,
                                                      summaryFunction=twoClassSummary,
                                                      classProbs=TRUE,
                                                      savePredictions=TRUE)))

modelo
#install.packages("pROC")
library("pROC")

g<-roc(modelo$pred$obs,
       modelo$pred$Si)
plot(g)

# proceso de seleccion de caracteristicas

install.packages("randomForest")
library("randomForest")

control <- rfeControl(functions = rfFuncs,
                      method="repeatedcv",
                      repeats=3,
                      verbose=TRUE)

respuesta <- 'Default'
predictores<-names(TRAIN)[!names(TRAIN) %in% respuesta]

seleccion <- rfe(TRAIN[,predictores],
                 TRAIN[,respuesta],
                 rfeControl = control)

seleccion
seleccion$optsize
seleccion$optVariables
seleccion$fit

plot(seleccion, type="b")