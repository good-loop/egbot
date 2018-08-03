package com.goodloop.egbot.tools;

import java.io.File;

import com.winterwell.utils.log.Log;

public class JsonFromCSV {

	public static void main(String[] args) {
		// all files
		File rawDataDir = new File("data/raw");
		File buildDataDir = new File("data/build");
		for(File f : rawDataDir.listFiles()) {
			if ( ! f.getName().endsWith(".csv")) continue;
			convertToJson(f);
		}
	}

	private static void convertToJson(File f) {
		Log.i("jsonfromcsv", "Converting "+f+" ...");
		foo
	}
}
