package com.goodloop.egbot.server;

import static org.junit.Assert.*;

import org.junit.Test;

public class MarkovModelTest {

	/**
	 * @Dan - TODO fix NPE
	 */
	@Test
	public void testLoad() {
		MarkovModel mm = new MarkovModel();
		mm.load();
	}

	@Test
	public void testSave() {
		fail("Not yet implemented");
	}

}
