import java.util.Stack;

/**
 * BinarySearchTree class the represents BinarySearchTree that contains the root of the Tree as field
 * @author oReL
 *
 */
public class BinarySearchTree {
	private Node root;
	public BinarySearchTree()
	{
		/**
		 * constructor 
		 */
	}
	public BinarySearchTree(int data)
	{
		/**
		 * constructor 
		 * @param int data that will be the data of the root
		 */
		this.root=new Node(data);//using the c'tor of Node class
	}
	public void Insert(int value)
	{
		/**
		 * Insert function just to execute insert process as we expected, like if there is BST obj ,so we will do obj.insert(value) and we'll not send also the obj for that as param
		 * @see insertHelp that will do the whole insert process
		 * @param int value of the node we want to add
		 */
		Node node=new Node(value);
		this.root=insertHelp(this.root,node);
	}
	private Node insertHelp(Node root,Node node)
	{
		/**
		 * insertHelp recursive function that will do the whole insert process
		 * @param Node root and Node node that will be the node we want to add
		 * @return return after the whole process the new root with the added node
		 */
		if(root==null)//if the root is null so we will make a new Node that will role as a root
		{
			root = new Node(node);
		}
		else if(root.getValue() > node.getValue())//if root data is bigger than the node data,so we go left
		{
			root.setLeft(insertHelp(root.getLeft(),node));
		}
		else if(node.getValue()> root.getValue())//if root data is smaller than the node data,so we go right
		{
			root.setRight(insertHelp(root.getRight(),node));
		}
		return root;
	}
	
	public Node Search(int value)
	{
		/**
		 * Search function just to execute search process as we expected, like if there is BST obj ,so we will do obj.search(value) and we'll not send also the obj for that as param
		 * @see searcheHelp that will do the whole search process
		 * @param the value of the Node we want to search
		 * @return Node node we wanted to search with the value we got as a param
		 */
		return searchHelp(this.root,value);
	}
	private Node searchHelp(Node root,int value)
	{
		/**
		 * searchHelp recursive function that will do the whole search process
		 * @param Node root and int value of the node that will be the node we want to search
		 * @return return after the whole process the node we searched for(or null if there's no BST)
		 */
		Node current = root;
		if (current ==null || current.getValue() == value)//if that the root or there is no BST,so it will return null
		{
			return current;
		}
		else if (value < current.getValue())//if value is smaller that the current data ,we go left
		{
			return searchHelp(current.getLeft(), value);
		}
		else//if value is bigger that the current data ,we go right
		{
			return searchHelp(current.getRight(), value);
		}
	}
	public boolean Delete(int value)
	{
		/**
		 * Delete function just to execute delete process as we expected, like if there is BST obj ,so we will do obj.delete(value) and we'll not send also the obj for that as param
		 * @see deleteHelp that will do the whole delete process
		 * @return returns true if node deleted and false otherwise
		 * @param the  value of the node we want to delete
		 */
		Node node=new Node(value);
		boolean isInTree=false;
		if(Search(node.getValue()) instanceof Node)//if that true it means the node is in the tree and gonna be deleted
			isInTree=true;
		this.root=deleteHelp(this.root,node);
		return isInTree;
	}
	private Node deleteHelp(Node root, Node node)
	{
		/**
		 * deleteHelp recursive function that will do the whole delete process
		 * @param Node root and Node node that will be the node we want to delete
		 * @return return the new root after the whole process with the deleted node
		 */
		if(root!=null)//if there is a root(that means there is bst) to delete from
		{
			if(node.getValue()< root.getValue())//if node.data is smaller than root data we go left
			{
				root.setLeft(deleteHelp(root.getLeft(),node));
			}
			else if(node.getValue()> root.getValue())//if node.data is bigger than root data we go right
			{
				root.setRight(deleteHelp(root.getRight(),node));
			}
			else if((root.getLeft()!=null) && (root.getRight()!=null))
			{
				Node minNode = new Node(root.getRight());
				Node right=minNode;
				while(minNode.getLeft()!=null)//find the successor and replace with the node we want to delete
				{
					minNode=minNode.getLeft();
				}
				root.setValue(minNode.getValue());
				right=deleteHelp(right,minNode);//delete the min node at the right sub tree
				minNode=null;
				root.setRight(right);
			}
			else
			{ 
				if(root.getLeft()!=null)
				{
					root=new Node(root.getLeft());
				}
				else
				{
					if(root.getRight()!=null)
						root=new Node(root.getRight());
					else
						root=null;
				}
			}
		}
		return root;
	}
	public void Iterative_inorder()
	{
		/**
		 * Iterative_inorder function just to execute iterative inorder process as we expected, like if there is BST obj ,so we will do obj.iterativeInorder() and we'll not send also the obj for that as param
		 * @see iterativeInorderHelp that will do the whole iterative inorder process
		 */
		iterativeInorderHelp(this.root);
	}
	private void iterativeInorderHelp(Node root)
	{
		/**
		 * iterativeInorderHelp  function that will do the whole iterative inorder process
		 * @param Node root
		 */
		//type Node
		Stack<Node> s= new Stack<Node>();
		while(!(s.isEmpty()) || (root !=null))//if the root isn't null so until the stack is empty we'll do the process
		{
			if(root !=null)//we go left
			{
				s.push(root);
				root = root.getLeft();
			}
			else//we can't go more left than that, so now we go right
			{
				root = s.pop();
				System.out.print(root.getValue()+" ");
				root = root.getRight();
			}
		}
		System.out.println();
	}
	public void Recursive_inorder()
	{
		/**
		 * Recursive_inorder function just to execute recursive inorder process as we expected, like if there is BST obj ,so we will do obj.recursiveInorder() and we'll not send also the obj for that as param
		 * @see recursiveInorderHelp that will do the whole recursive inorder process
		 */
		recursiveInorderHelp(this.root);
		System.out.println();
	}
	private void recursiveInorderHelp(Node root)
	{
		/**
		 * recursiveInorderHelp  function that will do the whole recursive inorder process
		 * @param Node root
		 */
		Node current = root;
		if(current !=null)
		{
			recursiveInorderHelp( current.getLeft());//go left as we can
			System.out.print(current.getValue()+" ");
			recursiveInorderHelp( current.getRight());//then go right
		}
	}
}

	
