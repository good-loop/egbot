package com.goodloop.egbot.server;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import com.winterwell.depot.Depot;
import com.winterwell.depot.Desc;
import com.winterwell.utils.IFilter;
import com.winterwell.utils.ReflectionUtils;
import com.winterwell.utils.log.Log;

public class ESModelTest {

//	@Test
	public void testModelWithMSE20Data() throws Exception {
		ESModel es = new ESModel();						
		new EvaluatePredictions().runModel(es, "MSE-20", "MSE-20", 1, 1, 1, "None", "None");
	}
	
//	@Test
	public void testModelWithPauliusData() throws Exception {
		ESModel es = new ESModel();						
		new EvaluatePredictions().runModel(es, "pauliusSample", "pauliusSample", 1, 1, 1, "None", "None");
	}
	
	@Test
	public void testScoreAnswer() throws Exception {
		ESModel es = new ESModel();				
		
		// get data files
		List<File> files = EgBotDataLoader.setup("pauliusSample");
		
		// init model
		es.init(files, 0, "", "");
		
		// index data 
		new EvaluatePredictions().runModel(es, "pauliusSample", "irinaSample", 1, 1, 1, "None", "None");			
		
		// score question
		// double score = es.scoreAnswer("Can you explain what a binomial distribution is?", "A binomial distribution model is a probability model and it is used when there are two possible outcomes");		
	}
}