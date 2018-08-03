package com.goodloop.egbot.tools;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;

import org.eclipse.jetty.util.ajax.JSON;

import com.winterwell.utils.Utils;
import com.winterwell.utils.containers.ArrayMap;
import com.winterwell.utils.io.CSVReader;
import com.winterwell.utils.io.CSVWriter;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.log.Log;
/**
 * Convert .csv to json - 
 * TODO Handling Python objects within the csv.
 * 
 * @author daniel
 *
 */
public class JsonFromCSV {

	public static void main(String[] args) throws IOException {
		// all files
		File rawDataDir = new File("data/raw");
		File buildDataDir = new File("data/build");
		buildDataDir.mkdirs();
		for(File f : rawDataDir.listFiles()) {
			if ( ! f.getName().endsWith(".csv")) continue;
			File fout = new File(buildDataDir, FileUtils.changeType(f, "json").getName());
			convertToJson(f, fout);
		}
	}

	private static void convertToJson(File f, File fout) throws IOException {
		Log.i("jsonfromcsv", "Converting "+f+" ...");
		CSVReader r = new CSVReader(f);
		BufferedWriter w = FileUtils.getWriter(fout);
		w.write("[\n	");
		String[] cols = r.next();
		boolean first = false;
		for (String[] row : r) {
			if ( ! first) w.write(",\n	");
			first = false;
			// make json obj
			ArrayMap jobj = new ArrayMap();
			for(int i=0; i<cols.length; i++) {
				if (i==row.length) break;
				String k = cols[i];
				if (Utils.isBlank(k)) k = "col_"+i;
				jobj.put(k, row[i]);
			}
			// write it out
			String json = JSON.toString(jobj);
			w.write(json);
		}
		w.write("\n]");
		r.close();
		w.close();
	}
}
