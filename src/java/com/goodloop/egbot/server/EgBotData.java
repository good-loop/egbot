package com.goodloop.egbot.server;

import java.io.File;
import java.util.List;

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

	public EgBotData(List<File> files, IFilter<Integer> filter) {
		super();
		this.files = files;
		this.filter = filter;
	}
	
	
}
