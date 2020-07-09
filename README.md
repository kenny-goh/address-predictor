# address-predictor

 AddressPredictor is a Java library that provides a predict function that classify
 the street, city, state and postcode from a given address text.
 
 The library contains the recurrent neural model (bidirectional + Long short term memory) that was trained on Keras (Python) with 30K rows of address data and can predict
 test data up to 99.8% accuracy.
 
 This library uses the DL4J library to load the model trained in Keras.
