package egbot;

import java.util.Arrays;
import java.util.Collection;

import com.winterwell.bob.BuildTask;
import com.winterwell.bob.tasks.MavenDependencyTask;

/**
 * Get jars etc for EgBot's java pieces
 * @author daniel
 *
 */
public class BuildEgBotJava extends BuildTask {
	@Override
	protected void doTask() throws Exception {
		MavenDependencyTask mdt = new MavenDependencyTask();
		mdt.addDependency("black.ninia","jep","3.8.2");
		mdt.addDependency("org.tensorflow", "tensorflow", "1.11.0");
		mdt.addDependency("org.deeplearning4j","deeplearning4j-core","1.0.0-beta2");
		mdt.addDependency("org.nd4j","nd4j-native-platform","1.0.0-beta2");
		mdt.addDependency("org.slf4j","slf4j-log4j12","1.8.0-beta2");
		mdt.run();
	}
}
