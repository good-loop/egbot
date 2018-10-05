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
		mdt.run();
	}
}
