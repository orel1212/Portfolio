/**
 *  Node class that represents a Node in Binary Search Tree  that contains fields as data,left and right
 * @author oReL
 *
 */
public class Node 
{
	private int data;
	private Node left;
	private Node right;
	public Node(int data)
	{
		/**
		 * constructor
		 * @param int data that will be the data of the node
		 */
		this.data=data;
		this.left=null;//if that new node,the left is null
		this.right=null;//also right
	}
	public Node(Node node)
	{
		/**
		 * copy constructor
		 * @param node that constains data,also maybe left and right and we want to copy that 
		 */
		this.data=node.data;
		this.left=node.left;
		this.right=node.right;
	}
	public void setValue(int data)
	{
		/**
		 * setData function that set a new data into data
		 * @param int data that will be the new data
		 */
		this.data=data;
	}
	public void setLeft(Node left)
	{
		/**
		 * setLeft function that set a new left into left
		 * @param Node left that will be the new left
		 */
		this.left=left;
	}
	public void setRight(Node right)
	{
		/**
		 * setRight function that set a new right into right
		 * @param Node right that will be the new right
		 */
		this.right=right;
	}
	public int getValue()
	{
		/**
		 * getData function that returns the current data
		 * @return the data 
		 */
		return this.data;
	}
	public Node getLeft()
	{
		/**
		 * getLeft function that returns the current left
		 * @return the left node 
		 */
		return this.left;
	}
	public Node getRight()
	{
		/**
		 * getRight function that returns the current right
		 * @return the right node 
		 */
		return this.right;
	}
	
}
