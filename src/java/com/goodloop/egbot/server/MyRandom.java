package com.goodloop.egbot.server;

import java.util.Random;

/**
 * ??use cases are weak - maybe use something else
 * 
 * fixed seed random number generator
 * @author Irina
 * @testedby {@link MyRandomTest}
 */
public class MyRandom {
	Random c;
	
	MyRandom() {
		c = new Random();
		c.setSeed(42);
	}

	public Random getC() {
		return c;
	}

}
