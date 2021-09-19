/**
 * interface of abstractseafactory
 * @author oReL
 *
 */
public interface AbstractSeaFactory
{
	/**
	 * produce a new seacreature
	 * @param string type
	 * @return SeaCreature by the type of the string
	 */
	public SeaCreature produceSeaCreature(String type);
}
