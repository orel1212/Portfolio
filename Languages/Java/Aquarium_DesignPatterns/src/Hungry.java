import javax.swing.JOptionPane;

/**
 * Hungry that implements HungerState interface
 * @author oReL
 *
 */
public class Hungry implements HungerState
{
	/**
	 * Implemenation of doAction from HungerState
	 * @see HungerState for more info
	 */
	@Override
	public void doAction(Swimmable s)
	{
		s.setState(this);
		
	}
}
