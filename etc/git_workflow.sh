git config --global user.name "James Elliott"
git config --global user.email "jjellio@sandia.gov"

git config --global color.ui auto
git config --global core.whitespace trailing-space

if grep -q "^source /etc/bash_completion.d/git$" ~/.bashrc; then 
  echo Git completion already set;
else
  echo "source /etc/bash_completion.d/git" >> ~/.bashrc

  source /etc/bash_completion.d/git
fi

