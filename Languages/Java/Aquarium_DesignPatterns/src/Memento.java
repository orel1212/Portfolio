import java.awt.Color;

/**
 * class that represent Memento to save the stats
 * @author oReL
 *
 */
public class Memento 
{
	private SeaCreatureState scs;
	/**
	 * c'tor
	 * @param scs that is the State info of the SeaCreature
	 */
	public Memento(SeaCreatureState scs)
	{
		this.scs=scs;
	}
	/**
	 * function that return the state
	 * @return scs state
	 */
	public SeaCreatureState getState()
	{
		return scs;
	}
}