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
	public void testMSEPart1ModelWithMSEPart1Data() throws Exception {
		ESModel es = new ESModel();
		new EvaluatePredictions().runModel(es, "MSE-part1", "MSE-part1", 100, 100, 1, "None", "None");
	}	
	
//	@Test
	public void testMSEFullModelWithMSEfullData() throws Exception {
		ESModel es = new ESModel();
		new EvaluatePredictions().runModel(es, "MSE-full", "MSE-full", 100, 100, 1, "None", "None");
	}	
	
//	@Test
	public void testMSEFullModelWithPauliusData() throws Exception {
		ESModel es = new ESModel();
		new EvaluatePredictions().runModel(es, "MSE-full", "pauliusSample", 1, 1, 1, "None", "None");
	}	
	
	@Test
	public void testMSE20ModelWithPaulius20Data() throws Exception {
		ESModel es = new ESModel();						
		new EvaluatePredictions().runModel(es, "MSE-20", "pauliusSample", 1, 1, 1, "None", "None");
	}
	
//	@Test
	public void testPauliusModelWithMSE20Data() throws Exception {
		ESModel es = new ESModel();						
		new EvaluatePredictions().runModel(es, "pauliusSample", "MSE-20", 1, 1, 1, "None", "None");
	}
	
//	@Test
	public void testScoreAnswer() throws Exception {
		ESModel es = new ESModel();
		new EvaluatePredictions().runModel(es, "pauliusSample", "irinaSample", 1, 1, 1, "None", "None");			
		
		// score question
		// double score = es.scoreAnswer("Can you explain what a binomial distribution is?", "A binomial distribution model is a probability model and it is used when there are two possible outcomes");		
	}
}