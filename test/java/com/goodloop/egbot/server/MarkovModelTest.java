package com.goodloop.egbot.server;

import org.junit.Test;

import com.winterwell.depot.Depot;

/**
 * smoke tests only so far
 * @author daniel
 *
 */
public class MarkovModelTest {

	@Test
	public void testLoad() {
		Depot.getDefault().init();
		MarkovModel mm = new MarkovModel();
		mm.load();
	}

	@Test
	public void testSave() {
		Depot.getDefault().init();
		MarkovModel mm = new MarkovModel();
		mm.save();
	}

}
