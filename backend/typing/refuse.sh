# Each file here but ok.ml is a program the types refuse, its first line
# the refusal it must get; ok.ml is one they accept. $1 compiles, $2 and $3
# hold the interfaces.
set -u
inc="-I $2 -I $3"
out=$(mktemp -d)
fail=0
squash() { tr '\n' ' ' | tr -s ' '; }
"$1" $inc -c -o "$out/ok" ok.ml || { echo "FAIL ok.ml is refused"; fail=1; }
for f in *.ml; do
  [ "$f" = ok.ml ] && continue
  want=$(head -1 "$f" | sed -e 's/^(\* refused: //' -e 's/ \*)$//' | tr -d '\n')
  if got=$("$1" $inc -c -o "$out/x" "$f" 2>&1); then echo "FAIL $f is accepted"; fail=1; continue; fi
  case "$(printf '%s' "$got" | squash)" in
    *"$want"*) ;;
    *) echo "FAIL $f: expected \"$want\""; printf '%s\n' "$got"; fail=1 ;;
  esac
done
rm -rf "$out"
[ $fail = 0 ] && echo "typing: every refusal is a type error"
exit $fail
