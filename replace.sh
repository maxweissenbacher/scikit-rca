old='project-template'
new='rca_fmri'

# Rename files and directories containing the term (depth-first)
find . -depth -name "*$old*" -print0 \
  | while IFS= read -r -d '' p; do
      dir=$(dirname "$p")
      base=$(basename "$p")
      newbase=${base//$old/$new}
      newpath="$dir/$newbase"
      if [ "$p" != "$newpath" ]; then
        mv -n -- "$p" "$newpath"
      fi
    done

