package com.gkh.deeplearning.address;

import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author Kenny Goh
 */
public class AddressPredictorTest {

	@Test
	public void verifyAll() throws AddressPredictionException {
		verify(null,"16 colville crescent","keysborough","3173","vic",true,",");
		verify("Plumbing John","32 queen road","Roxburg Park","4552","vic",true," ");
		verify(null,"PO BOX 32", "Tanah Merah", "3311", "QLD", true, ",");
		verify(null,"44 South Road", "Green Hill", "3311", "NSW", true, ",");
		verify("Dockland shopping centre","777 Hill Road", "Dockland", "3311", "Vic", true, "\n");
		verify(null,"777 Mining Road", "Sovereign Hill", "3345", "NT", true, "\n");
	}

	/**
	 *
	 * @param building optional, can be null
	 * @param street
	 * @param city
	 * @param postcode
	 * @param state
	 * @param randomlySwapCityStatePostcode Randomly swap position of city, state and postcode
	 * @param sep seperator
	 */
	public void verify(String building,
	                   String street,
	                   String city,
	                   String postcode,
	                   String state,
	                   boolean randomlySwapCityStatePostcode,
	                   String sep) throws AddressPredictionException {
		List<String> addressPart = new ArrayList<>();
		if (building != null) {
			addressPart.add(building);
		}
		addressPart.add(street);
		List<String> cityStatePostCode = new ArrayList<>();
		cityStatePostCode.add(city);
		cityStatePostCode.add(state);
		cityStatePostCode.add(postcode);
		if (randomlySwapCityStatePostcode) {
			Collections.shuffle(cityStatePostCode);
		}
		addressPart.addAll(cityStatePostCode);

		String addressText = String.join(sep, addressPart);
		System.out.println(addressText);

		Address address = AddressPredictor.predict(addressText);

		if (building != null) {
			Assert.assertEquals(building, address.getBuilding());
		}
		Assert.assertEquals(street, address.getStreet());
		Assert.assertEquals(state, address.getState());
		Assert.assertEquals(city, address.getCity());
		Assert.assertEquals(postcode, address.getPostcode());
	}


}
