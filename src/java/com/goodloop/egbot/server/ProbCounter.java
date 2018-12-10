package com.goodloop.egbot.server;

import java.util.Random;

public class ProbCounter {
	Random c;
	
	ProbCounter() {
		c = new Random();
		c.setSeed(42);
	}
}
