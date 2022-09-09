import java.awt.Color;

/**
 * class that represent the state of the seacreature
 * @author oReL
 *
 */
public class SeaCreatureState
{
	private Color col;
	private int size;
	private int x;
	private int y;
	private int verSpeed;
	private int horSpeed;
	private int index;
	private String type;
	/**
	 * c'tor
	 * @param type the type
	 * @param info of the seacreature
	 */
	public SeaCreatureState(String type,Object[] info)
	{
		this.type=type;
		this.col=(Color)info[0];
		size=(int) info[1];
		x=(int) info[2];
		y=(int) info[3];
		if(info.length==6)
		{
			horSpeed=(int) info[4];
			verSpeed=(int) info[5];
		}
	}
	/**
	 * return the name of the seacreature
	 * @return String name
	 */
	public String getName()
	{
		return type;
	}
	/**
	 * return the color of the seacreature
	 * @return Color color
	 */
	public Color getCol() {
		return col;
	}
	/**
	 * set the color of the seacreature
	 * @param Color color
	 */
	public void setCol(Color col) {
		this.col = col;
	}
	/**
	 * return the size of the seacreature
	 * @return int size
	 */
	public int getSize() {
		return size;
	}
	/**
	 * set the size of the seacreature
	 * @param int size
	 */
	public void setSize(int size) {
		this.size = size;
	}
	/**
	 * return the x of the seacreature
	 * @return int x
	 */
	public int getX() 
	{
		return x;
	}
	/**
	 * set the x of the seacreature
	 * @param int x
	 */
	public void setX(int x_loc) 
	{
		this.x = x_loc;
	}
	/**
	 * return the y of the seacreature
	 * @return int y
	 */
	public int getY() 
	{
		return y;
	}
	/**
	 * set the y of the seacreature
	 * @param int y
	 */
	public void setY(int y_loc) {
		this.y = y_loc;
	}
	/**
	 * return the verSpeed of the seacreature
	 * @return int verSpeed
	 */
	public int getVerspeed() {
		return verSpeed;
	}
	/**
	 * set the verSpeed of the seacreature
	 * @param int verSpeed
	 */
	public void setVerspeed(int ver_speed) {
		this.verSpeed = ver_speed;
	}
	/**
	 * return the horSpeed of the seacreature
	 * @return int horSpeed
	 */
	public int getHorspeed() 
	{
		return horSpeed;
	}
	/**
	 * set the hor_speed of the seacreature
	 * @param int hor_speed
	 */
	public void setHorspeed(int hor_speed) {
		this.horSpeed = hor_speed;
	}
	/**
	 * return the index of the seacreature
	 * @return int index
	 */
	public int getIndex()
	{
		return index;
	}
	/**
	 * set the index of the seacreature at the arrays
	 * @param int index
	 */
	public void setIndex(int index)
	{
		this.index = index;
	}
	/**
	 * function that update the info of the seacreature via object[] info
	 * @param Object[] info
	 */
	public void UpdateInfo(Object[] info)
	{
		
		this.col=(Color)info[0];
		size=(int) info[1];
		x=(int) info[2];
		y=(int) info[3];
		if(info.length>4)
		{
			horSpeed=(int) info[4];
			verSpeed=(int) info[5];
		}
	}
	/**
	 * function that return the info of the seacreature via object[]
	 * @return Object[] info
	 */
	public Object[] getInfo()
	{
		Object[] info=new Object[6];
		info[0]=col;
		info[1]=size;
		info[2]=x;
		info[3]=y;
		info[4]=horSpeed;
		info[5]=verSpeed;
		return info;
	}
}
