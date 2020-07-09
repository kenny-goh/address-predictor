package com.gkh.deeplearning.address;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class Address {
	private String building;
	private String street;
	private String city;
	private String state;
	private String postcode;
}
