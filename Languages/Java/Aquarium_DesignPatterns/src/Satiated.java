/**
 * Satiated class that implements hungerstate interface
 * @author oReL
 *@see Hungerstate
 */
public class Satiated implements HungerState
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
