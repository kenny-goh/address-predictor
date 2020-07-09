package com.gkh.deeplearning.address;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * AddressPredictor is a static class that provides a predict function that classify
 * the street, city, state and postcode from a given address text.
 *
 * The BRNN LSTM neural model was trained on Keras (Python) with 30K rows of address data and can predict
 * test data up to 99.8% accuracy.
 *
 * This class uses the ND4J library to load the model trained in Keras.

 * @author Kenny Goh
 */
@Slf4j
public class AddressPredictor {

	final static String CHARS = "0123456789abcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r";

	private static final String BUILDING = "building";
	private static final String STREET = "street";
	private static final String CITY = "city";
	private static final String POSTCODE = "postcode";
	private static final String STATE = "state";
	private static final String BLANK = "blank";

	final static List<String> LABEL_CLASS = List.of(BUILDING, STREET, CITY, POSTCODE, STATE, BLANK);

	private static MultiLayerNetwork model;

	static {
		try {
			String modelFilePath = new ClassPathResource("model.h5").getFile().getPath();
			model = KerasModelImport.importKerasSequentialModelAndWeights(modelFilePath);
		} catch (Exception e) {
			log.error("Unable to load model", e);
		}
	}

	/**
	 * Predict the address based on the provided text
	 *
	 * @param featuresText Text that represents typical Australian address. eg PO BOX 32, Tanah Merah, QLD, 43431
	 * @return Address value object where building, street, city, state and postcode are inferred from the prediction.
	 * @throws AddressPredictionException
	 */
	public static Address predict(String featuresText) throws AddressPredictionException {

		if (model == null) {
			throw new AddressPredictionException("Unable to predict. Model is not initialized");
		}

		INDArray features = charsEncode(featuresText);
		INDArray featuresReshaped = features.reshape(1, features.length());

		long start = System.currentTimeMillis();
		INDArray result = model.output(featuresReshaped);
		long end = System.currentTimeMillis();

		log.info("predict time: {} ms ", (end-start));

		Map<String,String> addressParts = new HashMap<>();
		addressParts.put("building","");
		addressParts.put("street","");
		addressParts.put("city","");
		addressParts.put("state","");
		addressParts.put("postcode","");
		for (int i = 0; i < result.rows(); i++) {
			INDArray row = result.getRow(i);
			int labelIndex = row.argMax(0).toIntVector()[0];
			String cls = LABEL_CLASS.get(labelIndex);
			char predictedChar = featuresText.charAt(i);
			addressParts.put(cls, addressParts.get(cls) + predictedChar);
		}

		return new Address(addressParts.get("building").trim(),
				addressParts.get("street").trim(),
				addressParts.get("city").trim(),
				addressParts.get("state").trim(),
				addressParts.get("postcode").trim());
	}


	/**
	 * Helper function to encode text based on the vocabulary CHARS
	 */
	private static INDArray charsEncode(String text) {
		INDArray vector = Nd4j.zeros(DataType.INT,text.length());
		int i = 0;
		for (char c : text.toLowerCase().toCharArray()) {
			int index = CHARS.indexOf(c);
			vector.putScalar(new int[] {i++},index);
		}
		return vector;
	}

}

