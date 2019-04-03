package com.goodloop.egbot.server;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

public class MyRandomTest {

	@Test
	public void testGetC() {
		List list = new ArrayList<>();
		for(int i=0; i<10; i++) {
			String s = "";
			MyRandom counter = new MyRandom();
			for (int j = 0; j < 4; j++) {
				int wrongIdx = counter.getC().nextInt(100); 				
				s += wrongIdx+" ";
			}
			list.add(s);
		}
		System.out.println(list);
		assert ! list.get(0).equals(list.get(1));
	}

}
