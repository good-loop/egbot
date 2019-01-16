package com.goodloop.egbot.server;

import java.io.File;
import java.util.List;

import com.winterwell.datascience.Experiment;
import com.winterwell.depot.Desc;
import com.winterwell.utils.IFilter;

/**
 * Egbot data defining its files and the filter used for training and evaluation 
 * 
 * @author irina
 *
 */
public class EgBotData {

	final List<File> files;
	
	final IFilter<Integer> filter;
	
	String use; // test or train
	
	Desc<EgBotData> desc;

	public EgBotData(List<File> files, IFilter<Integer> filter) {
		super();
		this.files = files;
		this.filter = filter;
	}
	
	public void setType(String use) {
		this.use = use;
	}

	public Desc<EgBotData> getDesc() {		
		Desc temp = new Desc("EgBotData", EgBotData.class);
		temp.put("use", use);
		return this.desc;
	}	
	
}
