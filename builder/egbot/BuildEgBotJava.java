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
	public Collection<? extends BuildTask> getDependencies() {
		return Arrays.asList(
			new MavenDependencyTask()
				.addDependency("org.python","jython-standalone", "2.7.1")
//				.setForceUpdate(true)
				.setIncSrc(true)			
				);
	}
	
	@Override
	protected void doTask() throws Exception {
	}

	
}
