/**
 * class Fish which represent a Immobile in the aquarium
 * @author oReL
 *
 */
public abstract class Immobile implements SeaCreature
{
	String name;
	protected SeaCreatureState scs;
	public Immobile(String name)
	{
		this.name=name;
	}
	/**
	 * returns the SeaCreatureState of the immobile
	 * @return SeaCreatureState of the immobile
	 */
	public SeaCreatureState getSeaCreatureState()
	{
		return scs;
	}
	/**
	 * function that do callback from aquapanel
	 * @param AquaPanel a to do callback
	 */
	public abstract void setAquaPanel(AquaPanel a) ;
	/**
	 * function that return the info of the immobile via object[]
	 * @return Object[] info
	 */
	public abstract Object[] getInfo();
	/**
	 * return the name of the plant
	 * @return String name
	 */
	public String getPlantName()
	 {
		 return this.name;
	 }
	/**
	 * function that return the size of the immobile
		* @return int size
	*/
	abstract public int getSize();
	/**
	 * function that restore the info of the immobile via object[] info
	 * @param Object[] info
	 */
	public abstract void setInfo(Object[] info);
	/**
	 * function that make new memento to save
	 * @see Memento class
	 * @return Memento obj
	 */
	public Memento MakeMomento()
	{
		return new Memento(scs);
	}	
	
}
